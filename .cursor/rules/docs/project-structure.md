# Project Structure

## Root Layout

```
nim-api-adapter/
├── models/api/                  # Model adapters (the core of the project)
│   ├── llm/                     # LLM adapters + model configs
│   │   ├── base.py              # Shared LLM adapter (ModelAdapter class)
│   │   ├── meta/                # Meta models (llama family)
│   │   ├── microsoft/           # Microsoft models (phi family)
│   │   ├── mistralai/           # Mistral models
│   │   ├── nvidia/              # NVIDIA models
│   │   ├── google/              # Google models (gemma)
│   │   └── ...                  # Other providers
│   ├── vlm/                     # Vision-Language Model adapters
│   │   ├── base.py
│   │   ├── meta/
│   │   └── microsoft/
│   ├── embeddings/              # Embedding adapters
│   │   ├── base.py
│   │   ├── nvidia/
│   │   └── baai/
│   └── object_detection/        # Object Detection adapters
│       ├── base.py
│       ├── baidu_paddleocr/
│       ├── nv_yolox_page_elements_v1/
│       └── university_at_buffalo_cached/
├── tests/
│   ├── e2e_tests/api/           # E2E tests (mirror models/api/ structure)
│   ├── assets/                  # Test datasets and fixtures
│   │   ├── e2e_tests/datasets/  # Datasets used by E2E tests
│   │   └── unittests/           # Unit test fixtures
│   └── unittests/               # Unit tests per model
├── .dataloop.cfg                # Lists all DPK manifests for publishing
├── Dockerfile                   # Runner image build
├── dlpytest.py                  # E2E test runner (dtlpytest framework)
├── publish_test.py              # DPK publishing script
└── validate_dataloop_json.py    # Validates all dataloop.json consistency
```

## Key Relationships

- Each `models/api/<category>/<vendor>/<model>/dataloop.json` defines a DPK
- `.dataloop.cfg` lists all dataloop.json paths for batch publishing
- Each model's `dataloop.json` points to `models/api/<category>/base.py` as the adapter
- E2E tests in `tests/e2e_tests/api/` must mirror the model path structure exactly
- Test datasets in `tests/assets/e2e_tests/datasets/` are referenced by name in config.yaml

## Available Datasets (for tests)

| Name | Contents | Used by |
|---|---|---|
| `text_prompt_text_answer` | Text prompt JSON | LLM, Embeddings |
| `text_image_prompt_text_answer` | Image + text prompt JSON | VLM |
| `chart_image` | Chart image + ontology | Object Detection |
| `deplot_image_prompt_text_answer` | Chart image + image-only prompt | Deplot VLM |

## Runner Image

All models share: `gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.14`

## Integration

All models require the `dl-ngc-api-key` integration, mapped to the `NGC_API_KEY` environment variable.
