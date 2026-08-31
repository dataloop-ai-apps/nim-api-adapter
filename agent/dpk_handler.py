# DPK manifest generation via JSON templates
"""
Builds Dataloop manifests for API adapters (``models/api``) using
``agent/templates/*.manifest.json``.
"""
import copy
import json
import os

ENV = os.environ.get("ENV", "rc")

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

# -----------------------------------------------------------------------------
# Manifest from JSON templates
# -----------------------------------------------------------------------------
_MANIFEST_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_MANIFEST_TEMPLATE_CACHE: dict[str, dict] = {}


DEFAULT_API_ADAPTER_RUNNER_IMAGE = os.environ.get(
    "NIM_API_ADAPTER_RUNNER_IMAGE",
    "gcr.io/viewo-g/piper/agent/runner/apps/nim-api-adapter:0.3.43",
)


def _coerce_manifest_model_type(model_type) -> str:
    """Normalize model type to manifest keys in MODEL_TYPE_CONFIG (llm / vlm / embedding)."""
    if model_type is None:
        return "llm"
    raw = model_type.value if hasattr(model_type, "value") else str(model_type)
    mt = raw.lower()
    if mt == "vlm_video":
        return "vlm"
    return mt if mt in MODEL_TYPE_CONFIG else "llm"


def _load_manifest_template_json(filename: str) -> dict:
    if filename not in _MANIFEST_TEMPLATE_CACHE:
        path = os.path.join(_MANIFEST_TEMPLATE_DIR, filename)
        with open(path, encoding="utf-8") as f:
            _MANIFEST_TEMPLATE_CACHE[filename] = json.load(f)
    return copy.deepcopy(_MANIFEST_TEMPLATE_CACHE[filename])


def _deep_replace_placeholder_strings(obj: object, replacements: dict[str, str]) -> object:
    if isinstance(obj, str):
        s = obj
        for token, value in replacements.items():
            s = s.replace(token, value)
        return s
    if isinstance(obj, list):
        return [_deep_replace_placeholder_strings(x, replacements) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_placeholder_strings(v, replacements) for k, v in obj.items()}
    return obj


def create_nim_dpk_manifest_via_template(
    model_id: str,
    model_type,
    embeddings_size: int = None,
    license: str = None,
    runner_image: str = None,
) -> dict:
    """
    Build a public API adapter manifest from ``agent/templates/*.manifest.json``.

    Fills placeholder tokens in the template JSON with model-specific values.
    """
    result = {
        "status": "pending",
        "dpk_name": None,
        "manifest": None,
        "adapter_path": None,
        "adapter_code": None,
        "error": None,
    }
    dpk_name = model_to_dpk_name(model_id)
    result["dpk_name"] = dpk_name

    mt = _coerce_manifest_model_type(model_type)
    rim = runner_image or DEFAULT_API_ADAPTER_RUNNER_IMAGE
    version = get_dpk_version()
    adapter_rel_path = ADAPTER_MAPPING.get(mt, ADAPTER_MAPPING["llm"])
    adapter_path = os.path.join(REPO_ROOT, adapter_rel_path)
    result["adapter_path"] = adapter_path

    lic = (license or "").strip() or ""
    adapter_desc = f"NVIDIA NIM adapter for {model_id}"
    if lic:
        top_description = f"{adapter_desc}. License: {lic}"
        attr_license = lic
    else:
        top_description = adapter_desc
        attr_license = "Unknown"

    display_name = model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()
    provider = get_model_provider(model_id)

    try:
        if mt == "embedding":
            tmpl = _load_manifest_template_json("api_embedding.manifest.json")
        else:
            tmpl = _load_manifest_template_json("api_llm_vlm.manifest.json")

        repl = {
            "@@DPK_NAME@@": dpk_name,
            "@@DISPLAY_NAME@@": display_name,
            "@@MODEL_ID@@": model_id,
            "@@DESCRIPTION@@": top_description,
            "@@ADAPTER_DESC@@": adapter_desc,
            "@@VERSION@@": version,
            "@@GIT_TAG@@": version,
            "@@RUNNER_IMAGE@@": rim,
            "@@PROVIDER@@": provider,
            "@@LICENSE@@": attr_license,
        }
        if mt != "embedding":
            repl["@@GEN_AI_LABEL@@"] = MODEL_TYPE_CONFIG[mt]["gen_ai"]
            repl["@@MEDIA_TYPE@@"] = "Multi Modal" if mt == "vlm" else "Text"
            repl["@@ENTRY_POINT@@"] = ADAPTER_MAPPING[mt]
        manifest = _deep_replace_placeholder_strings(tmpl, repl)

        if mt == "embedding":
            dim = embeddings_size if embeddings_size is not None else 1024
            manifest["components"]["models"][0]["configuration"]["embeddings_size"] = int(dim)

        with open(adapter_path, "r", encoding="utf-8") as rf:
            result["adapter_code"] = rf.read()

        result["status"] = "success"
        result["manifest"] = manifest

    except OSError as e:
        result["status"] = "error"
        result["error"] = f"manifest template/build failed: {e}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    """
    Dry-run test of DPK manifest generation logic.
    Tests all argument building (config, attributes, paths) per model type.
    Run: python agent/dpk_handler.py
    """
    import pprint
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("DPK HANDLER DRY-RUN")
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

    print("\n" + "=" * 60)
    print("DPK HANDLER DRY-RUN COMPLETE")
    print("=" * 60)
