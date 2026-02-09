from openai import OpenAI
import dtlpy as dl
import logging
import os

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
    # OpenAI client appends /embeddings; server expects /v1/embeddings, so base must end with /v1
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
            self.use_nvidia_extra_body = False  # downloadable app rejects input_type/truncate
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
        

    def call_model_open_ai(self, text):
        kwargs = dict(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
        )
        if self.use_nvidia_extra_body:
            kwargs["extra_body"] = {"input_type": "query", "truncate": "NONE"}
        response = self.client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding
        return embedding

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                text = item
            else:
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][0]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][0]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            embedding = self.call_model_open_ai(text)
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    print(f"Nim Model name: {model.configuration.get('nim_model_name')}")
    
    item = dl.items.get(item_id="ITEM_ID_HERE")
    adapter = ModelAdapter(model)
    result = adapter.embed_items([item])
    
    print(f"Embedding dimension: {len(result[0]) if result else 'N/A'}")