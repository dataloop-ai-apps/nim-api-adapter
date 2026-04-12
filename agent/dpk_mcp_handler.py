# DPK Generator MCP Client
"""
Client for generating DPK manifests for NVIDIA NIM models.

Calls MCP tools directly with explicit parameters (no LLM interpretation).
"""
import asyncio
import json
import os
import requests
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MCP transport mode (auto-selected):
#   DPK_MCP_NAME set → HTTP (platform-hosted DPK service on Dataloop)
#   MCP_SERVER_PATH set → stdio (local subprocess)
DPK_MCP_NAME = os.environ.get("DPK_MCP_NAME")
ENV = os.environ.get("ENV", "prod")
PYTHON_PATH = os.environ.get("PYTHON_PATH", "python")
MCP_SERVER_PATH = os.environ.get("MCP_SERVER_PATH")

def ensure_dataloop_login():
    """Ensure a valid Dataloop session exists, logging in if needed."""
    import dtlpy as dl
    dl.setenv(ENV)
    if dl.token_expired() or not dl.token():
        dl.login_m2m(
            email=os.environ.get("BOT_EMAIL"),
            password=os.environ.get("BOT_PASSWORD"),
        )


# Adapter paths mapping - relative to repo root (models/api/ folder)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_MAPPING = {
    "embedding": "models/api/embeddings/base.py",
    "vlm": "models/api/vlm/base.py",
    "llm": "models/api/llm/base.py"
}

_FALLBACK_VERSION = "0.3.53"
_dpk_version_cache: str | None = None


def get_dpk_version(use_github: bool = True) -> str:
    """
    Get the current DPK version from .bumpversion.cfg.

    Cached after first call. Strategy:
    1. GitHub API — read from **main** branch (remote, no local pull).
    2. Local .bumpversion.cfg on disk.
    3. Hardcoded fallback.
    """
    global _dpk_version_cache
    if _dpk_version_cache is not None:
        return _dpk_version_cache

    if use_github:
        try:
            from github_client import GitHubClient
            gh = GitHubClient()
            content = gh._get_file_content(".bumpversion.cfg", branch="main")
            if content:
                for line in content.splitlines():
                    if line.strip().startswith("current_version"):
                        version = line.split("=", 1)[1].strip()
                        print(f"  DPK version (GitHub main): {version}")
                        _dpk_version_cache = version
                        return version
        except Exception:
            pass

    try:
        bump_path = os.path.join(REPO_ROOT, ".bumpversion.cfg")
        with open(bump_path, "r") as f:
            for line in f:
                if line.strip().startswith("current_version"):
                    version = line.split("=", 1)[1].strip()
                    print(f"  DPK version (local .bumpversion.cfg): {version}")
                    _dpk_version_cache = version
                    return version
    except Exception:
        pass

    print(f"  DPK version: fallback {_FALLBACK_VERSION}")
    _dpk_version_cache = _FALLBACK_VERSION
    return _FALLBACK_VERSION


# =========================================================================
# Shared model utilities — imported by github_client, tester, nim_agent
# =========================================================================

# Model type to folder mapping (used for manifest paths in models/api/)
MODEL_TYPE_FOLDERS = {
    "embedding": "embeddings",
    "llm": "llm",
    "vlm": "vlm",
    "vlm_video": "vlm",
    "object_detection": "object_detection",
    "ocr": "ocr",
}


def parse_model_id(model_id: str) -> tuple[str, str]:
    """
    Parse a model ID into (publisher, model_name) with normalized casing.

    Examples:
        "nvidia/llama-3.1-70b-instruct" -> ("nvidia", "llama_3_1_70b_instruct")
        "meta/llama-3-8b"               -> ("meta", "llama_3_8b")
        "nv-embed-v1"                   -> ("nvidia", "nv_embed_v1")
    """
    if "/" in model_id:
        parts = model_id.split("/", 1)
        publisher = parts[0].lower().replace("-", "_")
        model_name = parts[1].lower().replace(".", "_").replace("-", "_")
    else:
        publisher = "nvidia"
        model_name = model_id.lower().replace(".", "_").replace("-", "_")
    return publisher, model_name


def model_to_dpk_name(model_id: str) -> str:
    """Convert model ID to DPK name.  e.g. "nvidia/llama-3.1-8b" -> "nim-llama-3-1-8b"."""
    name = model_id.split("/")[-1]
    name = name.replace(".", "-").replace("_", "-").lower()
    return f"nim-{name}"


def get_model_provider(model_id: str) -> str:
    """
    Extract formatted provider name from model_id.

    Examples: "meta/llama" -> "Meta", "nvidia/nv-embed" -> "NVIDIA"
    """
    provider_map = {
        # Core model providers
        "meta": "Meta",
        "nvidia": "NVIDIA",
        "mistralai": "MistralAI",
        "mistral": "MistralAI",
        "openai": "Open AI",
        "google": "Google",
        "microsoft": "Microsoft",
        "ibm": "IBM",
        "ai21": "AI21",
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "bigcode": "BigCode",

        # Infra / cloud
        "aws": "AWS",
        "databricks": "Databricks",
        "snowflake": "Snowflake",
        "mongodb": "MongoDB",
        "couchbase": "Couchbase",
        "singlestore": "SingleStore",
        "core42": "Core42",
        "dell": "Dell",

        # Frameworks / OSS
        "huggingface": "Hugging Face",
        "hugging_face": "Hugging Face",
        "langchain": "LangChain",
        "llamaindex": "LlamaIndex",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "openmmlab": "OpenMMLab",
        "opencv": "OpenCV",
        "ultralytics": "Ultralytics",
        "roboflow": "Roboflow",

        # Hardware
        "intel": "Intel",
        "amd": "AMD",
        "qualcomm": "Qualcomm",

        # Other
        "getty": "Getty Images",
        "gettyimages": "Getty Images",
        "dataloop": "Dataloop",
        "other": "Other"
    }

    if "/" in model_id:
        provider_raw = model_id.split("/")[0].lower()
    else:
        provider_raw = "nvidia"
    return provider_map.get(provider_raw, provider_raw.replace("-", " ").replace("_", " ").title())


def get_adapter_path(model_type: str) -> str:
    """Get the absolute adapter file path for a model type."""
    adapter_rel_path = ADAPTER_MAPPING.get(model_type, "models/api/llm/base.py")
    return os.path.join(REPO_ROOT, adapter_rel_path)


def get_model_folder(model_id: str, model_type: str) -> str:
    """
    Get the relative folder path for a model.

    Returns e.g. "models/api/llm/nvidia/llama_3_1_70b_instruct"
    """
    type_folder = MODEL_TYPE_FOLDERS.get(model_type, "llm")
    publisher, model_name = parse_model_id(model_id)
    return f"models/api/{type_folder}/{publisher}/{model_name}"


def get_manifest_path(model_id: str, model_type: str) -> str:
    """Get the relative path to dataloop.json for a model."""
    return f"{get_model_folder(model_id, model_type)}/dataloop.json"


# =========================================================================
# DPK manifest configuration
# =========================================================================

# Model type to DPK category/type mapping
# Includes attributes for the manifest
MODEL_TYPE_CONFIG = {
    "llm": {
        "model_category": "Gen AI",
        "model_type": "LLM",
        "input_type": "text",
        "output_type": "text",
        # Attributes
        "media_type": ["Text"],
        "gen_ai": "LLM",
        "nlp": "Conversational"
    },
    "vlm": {
        "model_category": "Gen AI",
        "model_type": "LMM",
        "input_type": "image",
        "output_type": "text",
        # Attributes
        "media_type": ["Multi Modal"],
        "gen_ai": "LMM",
        "nlp": "Conversational"
    },
    "embedding": {
        "model_category": "NLP",
        "model_type": "Embeddings",
        "input_type": "text",
        "output_type": "embedding",
        # Attributes (no Gen AI for embeddings)
        "media_type": ["Text"],
        "nlp": "Embeddings"
    }
}


class DPKGeneratorClient:
    """
    MCP Client for generating DPK manifests for NVIDIA NIM models.

    Supports two transport modes (auto-selected by env vars):
      - HTTP: DPK_MCP_NAME is set → resolves the service URL from the Dataloop platform
      - stdio: MCP_SERVER_PATH is set → spawns a local MCP server subprocess

    Raises ValueError if neither is configured.
    """

    def __init__(self):
        if DPK_MCP_NAME:
            self._mode = "http"
            self._server_url = self._resolve_dpk_url(DPK_MCP_NAME)
        elif MCP_SERVER_PATH:
            self._mode = "stdio"
        else:
            raise ValueError(
                "Set DPK_MCP_NAME (platform-hosted) or MCP_SERVER_PATH (local stdio)."
            )

    @staticmethod
    def _resolve_dpk_url(dpk_name: str) -> str:
        """Resolve the MCP service URL from a deployed Dataloop DPK app."""
        import dtlpy as dl
        ensure_dataloop_login()
        dpk = dl.dpks.get(dpk_name=dpk_name)
        filters = dl.Filters(field='dpkName', values=dpk.name, resource='apps')
        for app in dl.apps.list(filters=filters).all():
            app_url = app.routes['mcp']
            resp = requests.get(app_url, headers=dl.client_api.auth, allow_redirects=True)
            return resp.url
        raise ValueError(f"No running app found for DPK: {dpk_name}")

    def _http_call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool via HTTP JSON-RPC (platform-hosted mode)."""
        import dtlpy as dl
        token = dl.token()
        with httpx.Client(
            headers={"authorization": f"Bearer {token}", "x-dl-info": token},
            timeout=300,
        ) as client:
            def _jsonrpc(method, params=None):
                resp = client.post(
                    self._server_url,
                    json={"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1},
                    headers={"Accept": "text/event-stream, application/json"},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"MCP HTTP error {resp.status_code}: {resp.text}")
                if 'text/event-stream' in resp.headers.get('content-type', ''):
                    for line in resp.text.split('\n'):
                        if line.startswith('data: '):
                            data = json.loads(line[6:])
                            if 'result' in data:
                                return data['result']
                    return None
                return resp.json().get('result')

            _jsonrpc("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "nim-agent", "version": "1.0.0"},
            })
            result = _jsonrpc("tools/call", {"name": tool_name, "arguments": arguments})
            if result is None:
                raise RuntimeError("MCP tool returned no result")
            return json.loads(result['content'][0]['text'])

    async def _stdio_call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool via stdio subprocess (local mode)."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        server_params = StdioServerParameters(
            command=PYTHON_PATH,
            args=[MCP_SERVER_PATH],
            env={**os.environ, "NGC_API_KEY": os.environ.get("NGC_API_KEY", ""), "MCP_TRANSPORT": "stdio"},
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.isError:
                    raise RuntimeError(f"MCP tool error: {result.content}")
                if not result.content or not result.content[0].text:
                    raise RuntimeError(f"MCP tool returned empty response. Raw result: {result}")
                return json.loads(result.content[0].text)

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch to HTTP or stdio based on configured mode."""
        if self._mode == "http":
            return self._http_call_tool(tool_name, arguments)
        return asyncio.run(self._stdio_call_tool(tool_name, arguments))
    
    def create_nim_dpk_manifest(self, model_id: str, model_type: str, embeddings_size: int = None, license: str = None) -> dict:
        """
        Create a DPK manifest for a NVIDIA NIM model.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            model_type: Type of model ("llm", "vlm", "embedding")
            embeddings_size: Embedding dimension (only for embedding type, default 1024)
            license: Canonical Dataloop license name (e.g., "MIT", "Apache 2.0")
            
        Returns:
            dict with status, dpk_name, manifest, adapter_path, error
        """
        result = {
            "status": "pending",
            "dpk_name": None,
            "manifest": None,
            "adapter_path": None,
            "adapter_code": None,
            "error": None
        }
        
        dpk_name = model_to_dpk_name(model_id)
        result["dpk_name"] = dpk_name
        
        try:
            # Get adapter path
            adapter_rel_path = ADAPTER_MAPPING.get(model_type, "models/api/llm/base.py")
            adapter_path = os.path.join(REPO_ROOT, adapter_rel_path)
            result["adapter_path"] = adapter_path
            
            # Read adapter code
            with open(adapter_path, 'r') as f:
                result["adapter_code"] = f.read()
            
            # Get type config
            type_config = MODEL_TYPE_CONFIG.get(model_type, MODEL_TYPE_CONFIG["llm"])
            
            # Build display name from model_id
            display_name = model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()
            
            # Extract provider from model_id (e.g., "meta/llama" → "Meta")
            model_provider = get_model_provider(model_id)
            
            # Build attributes based on model type
            attributes = {
                "Hub": ["Nvidia", "Dataloop"],
                "Provider": model_provider,
                "Deployed By": "NVIDIA",
                "Category": ["Model", "NIM"],
                "Media Type": type_config["media_type"],
                "NLP": type_config["nlp"]
            }
            # Add "Gen AI" only for LLM/VLM (not for embeddings)
            if "gen_ai" in type_config:
                attributes["Gen AI"] = type_config["gen_ai"]
            
            if license:
                attributes["License"] = license
            
            # Build model configuration based on type
            if model_type == "embedding":
                model_configuration = {
                    "nim_model_name": model_id,
                    "embeddings_size": embeddings_size or 1024,
                    "hyde_model_name": "",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                }
            else:
                # LLM and VLM share the same config structure
                model_configuration = {
                    "nim_model_name": model_id,
                    "max_tokens": 512,
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "stream": True,
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "system_prompt": "You are a helpful and a bit cynical assistant. Give relevant and short answers, if you dont know the answer just say it, dont make up an answer",
                    "add_metadata": ["system.document.source"],
                }

            # Resolve version from main branch (cached after first call)
            version = get_dpk_version()

            # Build MCP tool arguments
            mcp_args = {
                "name": dpk_name,
                "display_name": f"{display_name}",
                "description": f"NVIDIA NIM adapter for {model_id}",
                "version": version,
                "scope": "public",
                "model_category": type_config["model_category"],
                "model_type": type_config["model_type"],
                "provider": model_provider,
                "trainable": False,
                "entry_point": adapter_rel_path,
                "class_name": "ModelAdapter",
                "runner_image": "gcr.io/viewo-g/piper/agent/runner/apps/nim-api-adapter:0.3.43",
                "integrations": ["dl-ngc-api-key"],
                "hub": ["Nvidia", "Dataloop"],
                "git_url": "https://github.com/dataloop-ai-apps/nim-api-adapter",
                "git_tag": version,
                "configuration": model_configuration,
                "attributes": attributes,
            }

            # inputType/outputType only for embeddings
            if model_type == "embedding":
                mcp_args["input_type"] = "text"

            # Both HTTP and stdio call "create_model_adapter" which returns {"success": bool, "manifest": {...}}
            tool_response = self._call_tool("create_model_adapter", mcp_args)
            if not tool_response.get("success"):
                raise RuntimeError(tool_response.get("error", "MCP tool returned failure"))
            manifest = tool_response["manifest"]

            result.update({
                "status": "success",
                "manifest": manifest
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result.update({
                "status": "error",
                "error": str(e)
            })
        
        return result
    

# =========================================================================
# Module-level convenience functions
# =========================================================================

def create_nim_manifest(model_id: str, model_type: str, embeddings_size: int = None) -> dict:
    """Create NIM DPK manifest (module-level convenience function)."""
    client = DPKGeneratorClient()
    return client.create_nim_dpk_manifest(model_id, model_type, embeddings_size=embeddings_size)


if __name__ == "__main__":
    """
    Dry-run test of DPK manifest generation logic.
    Tests all argument building (config, attributes, paths) per model type.
    If MCP_SERVER_PATH is set, also calls the real MCP tool.
    Run: python agent/dpk_mcp_handler.py
    """
    import pprint

    print("=" * 60)
    print("DPK MCP HANDLER DRY-RUN")
    print("=" * 60)

    # --- 0. Version resolution ---
    print("\n" + "-" * 60)
    print("0. get_dpk_version (from GitHub main -> local -> fallback)")
    print("-" * 60)
    version = get_dpk_version()
    print(f"  Resolved version: {version}")

    TEST_MODELS = [
        ("meta/llama-3.1-8b-instruct", "llm", None),
        ("meta/llama-3.2-11b-vision-instruct", "vlm", None),
        ("baai/bge-m3", "embedding", 1024),
        ("nvidia/nv-embed-v1", "embedding", 4096),
    ]

    # ------------------------------------------------------------------
    # 1. Pure logic tests (no MCP server needed)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. model_to_dpk_name")
    print("-" * 60)
    for model_id, mtype, _ in TEST_MODELS:
        dpk = model_to_dpk_name(model_id)
        print(f"  {model_id:50s} -> {dpk}")

    print("\n" + "-" * 60)
    print("2. get_model_provider")
    print("-" * 60)
    for model_id, mtype, _ in TEST_MODELS:
        provider = get_model_provider(model_id)
        print(f"  {model_id:50s} -> {provider}")

    print("\n" + "-" * 60)
    print("3. get_adapter_path + parse_model_id + get_manifest_path")
    print("-" * 60)
    for mtype in ("llm", "vlm", "embedding"):
        path = get_adapter_path(mtype)
        print(f"  {mtype:12s} -> {path}")
    for model_id, mtype, _ in TEST_MODELS:
        pub, name = parse_model_id(model_id)
        mpath = get_manifest_path(model_id, mtype)
        print(f"  {model_id:50s} -> pub={pub}, name={name}, manifest={mpath}")

    print("\n" + "-" * 60)
    print("4. Model configuration per type")
    print("-" * 60)
    for model_id, mtype, emb_size in TEST_MODELS:
        type_config = MODEL_TYPE_CONFIG.get(mtype, MODEL_TYPE_CONFIG["llm"])
        dpk_name = model_to_dpk_name(model_id)
        provider = get_model_provider(model_id)
        adapter_rel_path = ADAPTER_MAPPING.get(mtype, "models/api/llm/base.py")

        # Build configuration exactly as create_nim_dpk_manifest does
        if mtype == "embedding":
            config = {
                "nim_model_name": model_id,
                "embeddings_size": emb_size or 1024,
                "hyde_model_name": "",
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        else:
            config = {
                "nim_model_name": model_id,
                "max_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.7,
                "stream": True,
                "base_url": "https://integrate.api.nvidia.com/v1",
                "system_prompt": "You are a helpful and a bit cynical assistant. Give relevant and short answers, if you dont know the answer just say it, dont make up an answer",
                "add_metadata": ["system.document.source"],
            }

        attributes = {
            "Hub": ["Nvidia", "Dataloop"],
            "Provider": provider,
            "Deployed By": "NVIDIA",
            "Category": ["Model", "NIM"],
            "Media Type": type_config["media_type"],
            "NLP": type_config["nlp"],
        }
        if "gen_ai" in type_config:
            attributes["Gen AI"] = type_config["gen_ai"]

        has_input_type = mtype == "embedding"

        print(f"\n  [{mtype.upper()}] {model_id}")
        print(f"    dpk_name:     {dpk_name}")
        print(f"    provider:     {provider}")
        print(f"    adapter:      {adapter_rel_path}")
        print(f"    inputType:    {'text' if has_input_type else '(none)'}")
        print(f"    outputType:   {type_config['output_type'] if has_input_type else '(none)'}")
        print(f"    configuration:")
        for k, v in config.items():
            val = repr(v) if isinstance(v, str) and len(v) > 40 else v
            print(f"      {k}: {val}")
        print(f"    attributes:")
        for k, v in attributes.items():
            print(f"      {k}: {v}")

    # ------------------------------------------------------------------
    # 5. Full MCP call (HTTP via DPK_MCP_NAME, or stdio via MCP_SERVER_PATH)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("5. Full MCP manifest creation")
    print("-" * 60)

    if DPK_MCP_NAME:
        print(f"  Mode: HTTP (DPK_MCP_NAME={DPK_MCP_NAME})")
    elif MCP_SERVER_PATH:
        print(f"  Mode: stdio (MCP_SERVER_PATH={MCP_SERVER_PATH})")
    else:
        print("  Skipping: set DPK_MCP_NAME (HTTP) or MCP_SERVER_PATH (stdio)")

    if DPK_MCP_NAME or MCP_SERVER_PATH:
        for model_id, mtype, emb_size in TEST_MODELS:
            print(f"\n  >> {model_id} ({mtype})")
            result = create_nim_manifest(model_id, mtype, embeddings_size=emb_size)
            print(f"     status:   {result['status']}")
            print(f"     dpk_name: {result['dpk_name']}")
            if result["error"]:
                print(f"     error:    {result['error'][:120]}")
            if result["manifest"]:
                # Show just the models[0].configuration from the generated manifest
                print(f"     manifest: {json.dumps(result['manifest'], indent=2)}")

    print("\n" + "=" * 60)
    print("DPK MCP HANDLER DRY-RUN COMPLETE")
    print("=" * 60)
