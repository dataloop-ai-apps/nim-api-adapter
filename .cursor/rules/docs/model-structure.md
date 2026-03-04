# Model & DPK Structure

## Directory Layout

```
models/api/<category>/<vendor>/<model_name>/
├── dataloop.json    # DPK manifest (single source of truth)
└── (no other files - adapter code is in base.py)
```

Adapter code lives at `models/api/<category>/base.py` — one per category, shared by all models.

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
  "name": "nim-llama-3-1-8b-instruct",       // DPK name (kebab-case)
  "displayName": "Llama 3.1 8B Instruct",    // Human-readable name
  "version": "0.0.14",                        // Package version
  "scope": "project",
  "components": {
    "computeConfigs": [...],                   // Runtime: podType, concurrency, autoscaler
    "modules": [{
      "entryPoint": "models/api/llm/base.py", // Category-specific adapter
      "className": "ModelAdapter",
      "integrations": ["dl-ngc-api-key"],      // Always requires NGC API key
      "functions": [...]
    }],
    "models": [{
      "name": "nim-llama-3-1-8b-instruct",    // Must match top-level name
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

## Categories and Adapters

| Category | Adapter Entry Point | Functions |
|---|---|---|
| `llm` | `models/api/llm/base.py` | predict_items, predict_dataset, evaluate_model |
| `vlm` | `models/api/vlm/base.py` | predict_items, predict_dataset, evaluate_model |
| `embeddings` | `models/api/embeddings/base.py` | predict_items, predict_dataset, evaluate_model |
| `object_detection` | `models/api/object_detection/base.py` | predict_items, predict_dataset |

## Validation

Run `python validate_dataloop_json.py` to check all `dataloop.json` files for:
- Model name ↔ module name consistency
- EntryPoint file existence and category matching
- Integration and runner image consistency
- Codebase configuration consistency
