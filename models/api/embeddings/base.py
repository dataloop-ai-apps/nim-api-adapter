from openai import OpenAI
import dtlpy as dl
import logging
import os
import httpx
import requests
import jwt
import datetime

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
    app = dl.apps.get(app_id=app_id)
    route = list(app.routes.values())[0].rstrip("/")
    base_before = "/".join(route.split("/")[:-1])
    session = requests.Session()
    resp = session.get(base_before, headers=dl.client_api.auth, verify=False)
    base_url = resp.url.rstrip("/")
    # OpenAI client appends /embeddings; server expects /v1/embeddings, so base must end with /v1
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return base_url, session.cookies, session


class ModelAdapter(dl.BaseModelAdapter):
    
    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

        self.app_id = self.configuration.get("app_id")
        if self.app_id:
            self.use_nvidia_extra_body = False  # downloadable app rejects input_type/truncate
            self.using_downloadable = True
            self.get_downloadable_client(self.app_id)
        else:
            self.using_downloadable = False
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
        
    def get_downloadable_client(self, app_id: str):
        self.base_url, cookies, self.current_session = get_downloadable_endpoint_and_cookie(app_id)
        cookie_header = "; ".join(f"{c.name}={c.value}" for c in cookies)

        logger.info(f"Using downloadable endpoint for {self.nim_model_name}, base URL: {self.base_url}")
        # Cookie-only auth: do not send Authorization or server returns "Multiple tokens provided"
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="",  # omit Bearer token so only Cookie header is sent
            default_headers={"Cookie": cookie_header},
            http_client=http_client
        )
        try:
            # Downloadable app exposes GET /v1/models (see app OpenAPI docs)
            health_url = self.base_url.rstrip("/") + "/manifest"
            r = requests.get(health_url, headers={"Cookie": cookie_header}, timeout=10, verify=False)
            r.raise_for_status()
            logger.info(f"Downloadable endpoint manifest for {self.nim_model_name}, base URL: {self.base_url}, response: {r.content}")
        except Exception as e:
            print(f"Health check failed: {e}")
    
    def check_jwt_expiration(self, margin_seconds: int = 60):
        """Check JWT expiration and refresh session if expired or about to expire."""
        token = self.current_session.cookies.get('JWT-APP')
        if not token:
            logger.warning("No JWT-APP cookie found, refreshing session")
            self.get_downloadable_client(self.app_id)
            return
        
        decoded = jwt.decode(token, options={"verify_signature": False})
        exp_timestamp = decoded.get('exp')
        if not exp_timestamp:
            logger.warning("No 'exp' claim in JWT, refreshing session")
            self.get_downloadable_client(self.app_id)
            return
        
        exp_dt = datetime.datetime.fromtimestamp(exp_timestamp)
        now = datetime.datetime.now()
        remaining = exp_dt - now
        
        if now >= exp_dt - datetime.timedelta(seconds=margin_seconds):
            logger.info(f"JWT expired or expiring soon (remaining: {remaining}). Refreshing session.")
            self.get_downloadable_client(self.app_id)
        else:
            logger.info(f"JWT still valid (remaining: {remaining})")
    
    def call_model_open_ai(self, text):
        kwargs = dict(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
        )
        if self.use_nvidia_extra_body:
            kwargs["extra_body"] = {"input_type": "query", "truncate": "NONE"}
        try:
            response = self.client.embeddings.create(**kwargs)
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Embeddings API call failed. Base URL: {self.base_url}, Model: {self.nim_model_name}, Error: {e}")
            raise

    def embed(self, batch, **kwargs):
        
        if self.using_downloadable:
            self.check_jwt_expiration()
            
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