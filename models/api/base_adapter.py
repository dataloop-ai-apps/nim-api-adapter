"""
NIM Base Model Adapter

Shared base class for all NVIDIA NIM model adapters (LLM, VLM, Embeddings).
Handles:
- API client setup (NVIDIA API or downloadable app)
- Downloadable endpoint resolution and JWT cookie auth
- JWT expiration checking and automatic session refresh
"""
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
        (base_url, cookies, session): base_url is the redirected API root;
        cookies is the session cookie jar; session is the requests.Session.
    """
    app = dl.apps.get(app_id=app_id)
    route = list(app.routes.values())[0].rstrip("/")
    base_before = "/".join(route.split("/")[:-1])
    session = requests.Session()
    resp = session.get(base_before, headers=dl.client_api.auth, verify=False)
    base_url = resp.url.rstrip("/")
    # OpenAI client appends /embeddings or /chat/completions; server expects /v1 prefix
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url, session.cookies, session


class NIMBaseAdapter(dl.BaseModelAdapter):
    """
    Base adapter for all NVIDIA NIM model types.

    Handles API/downloadable client setup and JWT session management.
    Subclasses must implement model-specific methods:
    - Embeddings: call_model_open_ai(), embed()
    - LLM/VLM: call_model(), predict(), prepare_item_func()
    """

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError(
                "Missing `nim_model_name` from model.configuration, "
                "cant load the model without it"
            )

        self.app_id = self.configuration.get("app_id")
        if self.app_id:
            self.use_nvidia_extra_body = False  # downloadable app rejects input_type/truncate
            self.using_downloadable = True
            self.get_downloadable_client(self.app_id)
        else:
            self.using_downloadable = False
            self.use_nvidia_extra_body = True
            self.base_url = self.configuration.get(
                "base_url", "https://integrate.api.nvidia.com/v1"
            )
            logger.info(f"Using base URL: {self.base_url}")
            self.api_key = os.environ.get("NGC_API_KEY")
            if not self.api_key:
                raise ValueError("Missing NGC_API_KEY environment variable")
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

            try:
                self.client.models.list()
                logger.info(
                    f"API key validated for {self.nim_model_name}, base URL: {self.base_url}"
                )
            except Exception as e:
                raise ValueError(f"API key validation failed: {e}")

    def get_downloadable_client(self, app_id: str):
        """Create OpenAI client authenticated via JWT cookie for downloadable NIM apps."""
        self.base_url, cookies, self.current_session = get_downloadable_endpoint_and_cookie(app_id)
        cookie_header = "; ".join(f"{c.name}={c.value}" for c in cookies)

        logger.info(
            f"Using downloadable endpoint for {self.nim_model_name}, base URL: {self.base_url}"
        )
        # Cookie-only auth: do not send Authorization or server returns "Multiple tokens provided"
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="",  # omit Bearer token so only Cookie header is sent
            default_headers={"Cookie": cookie_header},
            http_client=http_client,
        )
        try:
            health_url = self.base_url.rstrip("/") + "/manifest"
            r = requests.get(
                health_url, headers={"Cookie": cookie_header}, timeout=10, verify=False
            )
            r.raise_for_status()
            logger.info(
                f"Downloadable endpoint healthy for {self.nim_model_name}, "
                f"base URL: {self.base_url}"
            )
        except Exception as e:
            print(f"Health check failed: {e}")

    def check_jwt_expiration(self, margin_seconds: int = 60):
        """Check JWT expiration and refresh session if expired or about to expire."""
        token = self.current_session.cookies.get("JWT-APP")
        if not token:
            logger.warning("No JWT-APP cookie found, refreshing session")
            self.get_downloadable_client(self.app_id)
            return

        decoded = jwt.decode(token, options={"verify_signature": False})
        exp_timestamp = decoded.get("exp")
        if not exp_timestamp:
            logger.warning("No 'exp' claim in JWT, refreshing session")
            self.get_downloadable_client(self.app_id)
            return

        exp_dt = datetime.datetime.fromtimestamp(exp_timestamp)
        now = datetime.datetime.now()
        remaining = exp_dt - now

        if now >= exp_dt - datetime.timedelta(seconds=margin_seconds):
            logger.info(
                f"JWT expired or expiring soon (remaining: {remaining}). Refreshing session."
            )
            self.get_downloadable_client(self.app_id)
        else:
            logger.info(f"JWT still valid (remaining: {remaining})")
