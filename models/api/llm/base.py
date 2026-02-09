from openai import OpenAI
import dtlpy as dl
import logging
import time
import os
import json

# Toggleable logger - set NIM_DISABLE_LOGGING=1 to disable
if os.environ.get("NIM_DISABLE_LOGGING", "").lower() in ("1", "true", "yes"):
    logger = logging.getLogger("NIM Adapter")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
else:
    logger = logging.getLogger("NIM Adapter")

def get_downloadable_endpoint_and_cookie(app_id: str):
    """
    Resolve Dataloop app route and obtain JWT-APP cookie via redirect.
    Use when the model adapter should talk to a downloadable NIM app (.apps.dataloop.ai).

    Returns:
        (base_url, cookie_header): base_url is the redirected API root; cookie_header is the Cookie header value.
    """
    import requests
    app = dl.apps.get(app_id=app_id)
    route = list(app.routes.values())[0].rstrip("/")
    base_before = "/".join(route.split("/")[:-1])
    session = requests.Session()
    resp = session.get(base_before, headers=dl.client_api.auth)
    base_url = resp.url.rstrip("/")
    # OpenAI client uses /v1/chat/completions etc.; server expects /v1, so base must end with /v1
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    cookie_header = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    return base_url, cookie_header

class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

        app_id = self.configuration.get("app_id")
        if app_id:
            self.use_nvidia_extra_body = False  #  consistency with embeddings
            self.base_url, cookie_header = get_downloadable_endpoint_and_cookie(app_id)
            logger.info(f"Using downloadable endpoint for {self.nim_model_name}, base URL: {self.base_url}")
            # Cookie-only auth: do not send Authorization or server returns "Multiple tokens provided"
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="",  # omit Bearer token so only Cookie header is sent
                default_headers={"Cookie": cookie_header},
            )
            try:
                import requests
                # Downloadable app exposes GET /v1/health/live (see app OpenAPI docs)
                health_url = self.base_url.rstrip("/") + "/health/live"
                r = requests.get(health_url, headers={"Cookie": cookie_header}, timeout=10)
                r.raise_for_status()
                logger.info(f"Downloadable endpoint healthy for {self.nim_model_name}, base URL: {self.base_url}")
            except Exception as e:
                raise ValueError(f"Health check failed: {e}")
        else:
            self.use_nvidia_extra_body = True
            self.base_url = self.configuration.get("base_url", "https://integrate.api.nvidia.com/v1")
            logger.info(f"Using base URL: {self.base_url}")
            self.api_key = os.environ.get("NGC_API_KEY")
            if not self.api_key:
                raise ValueError("Missing NGC_API_KEY environment variable")
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            
            try:
                self.client.models.list()
                logger.info(f"API key validated for {self.nim_model_name}, base URL: {self.base_url}")
            except Exception as e:
                raise ValueError(f"API key validation failed: {e}")
        
        # Lower default to avoid context length issues on smaller models
        self.max_tokens = self.configuration.get('max_tokens', 512)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.7)
        self.seed = self.configuration.get('seed', None)
        self.stream = self.configuration.get('stream', False)
        self.guided_json = self.configuration.get("guided_json", None)
        self.debounce_interval = self.configuration.get('debounce_interval', 2)
        self.system_prompt = self.configuration.get('system_prompt', None)

        if self.guided_json is not None:
            try:
                item = dl.items.get(item_id=self.guided_json)
                binaries = item.download(save_locally=False)
                self.guided_json = json.loads(binaries.getvalue().decode("utf-8"))
                logger.info(f"Guided json: {self.guided_json}")
            except Exception as e:
                try:
                    self.guided_json = json.loads(self.guided_json)
                except Exception as e:
                    logger.error(f"Error loading guided json: {e}")


    def prepare_item_func(self, item: dl.Item):
        return dl.PromptItem.from_item(item=item)

    def _flatten_messages(self, messages: list[dict]) -> list[dict]:
        """
        Flatten message content from array format to plain string.
        
        Some NVIDIA NIM LLM models expect plain string content, not the 
        OpenAI multimodal format [{"type": "text", "text": "..."}].
        """
        flattened = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            if isinstance(content, list):
                # Extract text from array format
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif "text" in part:
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                flattened.append({"role": role, "content": " ".join(text_parts)})
            else:
                flattened.append({"role": role, "content": content})
        
        return flattened

    def call_model(self, messages: list[dict]):
        """Call NVIDIA NIM chat completions API."""
        # Flatten messages - LLMs expect plain string content, not multimodal arrays
        messages = self._flatten_messages(messages)
        
        extra_body = {}
        if self.guided_json and self.use_nvidia_extra_body:
            extra_body["guided_json"] = self.guided_json
            logger.info(f"Using guided_json: {self.guided_json}")

        # Build kwargs - omit seed as some models reject it
        kwargs = {
            "model": self.nim_model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self.client.chat.completions.create(**kwargs)

    def handle_response(self, prompt_item, response):
        """Handle streaming or non-streaming response."""
        if self.stream:
            full_response = ""
            last_update_time = time.time()

            for chunk in response:
                # Check if choices exists and has items
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta:
                    chunk_text = getattr(delta, 'content', "") or ""
                    if chunk_text:
                        full_response += chunk_text
                        if time.time() - last_update_time >= self.debounce_interval:
                            self.add_response(prompt_item, full_response)
                            last_update_time = time.time()

            self.add_response(prompt_item, full_response)
        else:
            # Check if choices exists and has items
            if not response.choices:
                logger.warning(f"Empty response from model {self.nim_model_name}")
                self.add_response(prompt_item, "")
                return
            message = getattr(response.choices[0], 'message', None)
            self.add_response(prompt_item, getattr(message, 'content', "") if message else "")

    def add_response(self, prompt_item, response_text):
        """Add response to prompt item."""
        if not response_text:
            return

        prompt_item.add(
            message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response_text}]},
            model_info={
                'name': self.model_entity.name,
                'confidence': 1.0,
                'model_id': self.model_entity.id,
            },
        )

    def predict(self, batch, **kwargs):
        """Run prediction on a batch of prompts."""
        for prompt_item in batch:
            # Get OpenAI-compatible messages directly
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            
            # Add system prompt if configured
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            response = self.call_model(messages)
            self.handle_response(prompt_item, response)

        return []

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    print(f"Nim Model name: {model.configuration.get('nim_model_name')}")
    
    item = dl.items.get(item_id="ITEM_ID_HERE")
    adapter = ModelAdapter(model)
    adapter.predict_items([item])