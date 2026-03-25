# Model & DPK Structure

## Directory Layout

```
models/api/<category>/<vendor>/<model_name>/
├── dataloop.json    # DPK manifest (single source of truth)
└── (no other files — adapter code is in base.py)
```

Each category has one shared adapter at `models/api/<category>/base.py`. All adapters inherit from `NIMBaseAdapter` in `models/api/base_adapter.py`.

## Naming Conventions

| Context | Format | Example |
|---|---|---|
| Folder name | `snake_case` | `llama_3_1_8b_instruct` |
| DPK name (dataloop.json `"name"`) | `kebab-case` with `nim-` prefix | `nim-llama-3-1-8b-instruct` |
| Model component name | Same as DPK name | `nim-llama-3-1-8b-instruct` |

**Rule**: The DPK name and the model component name inside `dataloop.json` are always identical. Never guess — always read `dataloop.json`.

## dataloop.json Key Fields

```json
{
  "name": "nim-llama-3-1-8b-instruct",
  "displayName": "Llama 3.1 8B Instruct",
  "version": "0.0.14",
  "scope": "project",
  "components": {
    "computeConfigs": [...],
    "modules": [{
      "entryPoint": "models/api/llm/base.py",
      "className": "ModelAdapter",
      "integrations": ["dl-ngc-api-key"],
      "functions": [...]
    }],
    "models": [{
      "name": "nim-llama-3-1-8b-instruct",
      "moduleName": "nim-llama-3-1-8b-instruct-module",
      "configuration": {
        "nim_model_name": "meta/llama-3.1-8b-instruct",
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
        "stream": true,
        "base_url": "https://integrate.api.nvidia.com/v1"
      }
    }]
  }
}
```

## Adapter Architecture

### Base Adapter (`models/api/base_adapter.py`)

`NIMBaseAdapter` handles:
- **API mode**: Creates OpenAI client with `NGC_API_KEY` against `base_url` (default: `https://integrate.api.nvidia.com/v1`)
- **Downloadable mode**: When `app_id` is in config, resolves Dataloop app route and authenticates via JWT cookie (no API key)
- Sets `use_nvidia_extra_body = True` for API mode, `False` for downloadable

### Category Adapters

| Category | Adapter Entry Point | Key Methods |
|---|---|---|
| `llm` | `models/api/llm/base.py` | `call_model()`, `predict()`, `_flatten_messages()` |
| `vlm` | `models/api/vlm/base.py` | `call_model()`, `predict()`, `prepare_item_func()` |
| `embeddings` | `models/api/embeddings/base.py` | `call_model_open_ai()`, `embed()` |
| `object_detection` | `models/api/object_detection/base.py` | `predict()` |

### LLM Structured Output (`guided_json`)

The LLM adapter supports NVIDIA's guided decoding via `guided_json` in model configuration:
- Reads `guided_json` from `self.configuration` (JSON string or dict)
- When `use_nvidia_extra_body` is `True` (API mode), sends it as `extra_body["nvext"]["guided_json"]`
- Downloadable endpoints do not support `nvext` extensions
- The NVIDIA hosted API may silently ignore `guided_json` for some models

### LLM Context Injection (RAG)

The LLM adapter supports RAG-style context injection:
- `nearestItems` from the prompt metadata are converted via `build_context()`
- Context is inserted as an assistant message before the last user message

## Validation

Run `python validate_dataloop_json.py` to check all `dataloop.json` files for:
- Model name / module name consistency
- EntryPoint file existence and category matching
- Integration and runner image consistency
- Codebase configuration consistency
