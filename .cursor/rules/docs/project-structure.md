# Project Structure

## Root Layout

```
nim-api-adapter/
├── models/
│   ├── api/                         # API models (96 DPKs) — call NVIDIA hosted endpoints
│   │   ├── base_adapter.py          # NIMBaseAdapter — shared base for all adapters
│   │   ├── llm/                     # 25 vendors, ~80 models
│   │   │   └── base.py              # Shared LLM adapter (ModelAdapter)
│   │   ├── vlm/                     # 3 vendors (meta, microsoft, nvidia)
│   │   │   └── base.py
│   │   ├── embeddings/              # 2 vendors (baai, nvidia)
│   │   │   └── base.py
│   │   └── object_detection/        # 3 models (baidu_paddleocr, nv_yolox, university_at_buffalo)
│   │       └── base.py
│   └── downloadable/                # Downloadable models — run on GPU via custom NIM runner
│       ├── main.py                  # Shared NIM runner (GPU start script)
│       ├── llm/, vlm/, embeddings/, object_detection/
│       └── tests/test_simple.py
├── agent/                           # NIM Agent — automated DPK creation & testing
│   ├── nim_agent.py                 # Agent orchestrator
│   ├── tester.py                    # Test automation
│   ├── github_client.py             # GitHub integration
│   ├── downloadables_create.py      # Downloadable DPK generator
│   ├── dpk_mcp_handler.py           # MCP handler for DPK operations
│   ├── manifest_template.json       # Template for new dataloop.json
│   ├── Dockerfile.template          # Template for new Dockerfiles
│   └── tests/test_agent.py
├── tests/
│   ├── e2e_tests/
│   │   ├── api/                     # E2E tests for API models (mirror models/api/ structure)
│   │   └── downloadable/            # E2E tests for downloadable models
│   ├── assets/
│   │   ├── e2e_tests/datasets/      # Test datasets (text_prompt, images, etc.)
│   │   └── unittests/               # Unit test fixtures
│   └── unittests/                   # Unit tests per model
├── .dataloop.cfg                    # Lists all DPK manifests (API + downloadable) for publishing
├── Dockerfile                       # Runner image build
├── dlpytest.py                      # E2E test runner (dtlpytest framework)
├── publish_test.py                  # DPK publishing helper script
└── validate_dataloop_json.py        # Validates all dataloop.json consistency
```

## LLM Vendors (models/api/llm/)

25 vendors: `abacusai`, `ai21labs`, `baichuan_inc`, `bytedance`, `google`, `gotocompany`, `ibm`, `igenius`, `institute_of_science_tokyo`, `meta`, `microsoft`, `minimaxai`, `mistralai`, `moonshotai`, `nvidia`, `openai`, `qwen`, `sarvamai`, `speakleash`, `stepfun_ai`, `thudm`, `tiiuae`, `tokyotech_llm`, `upstage`, `z_ai`

## VLM Vendors (models/api/vlm/)

3 vendors: `meta`, `microsoft`, `nvidia`

## Embeddings Vendors (models/api/embeddings/)

2 vendors: `baai`, `nvidia`

## Object Detection (models/api/object_detection/)

3 models: `baidu_paddleocr`, `nv_yolox_page_elements_v1`, `university_at_buffalo_cached`

## Key Relationships

- Each `models/api/<category>/<vendor>/<model>/dataloop.json` defines a DPK
- `.dataloop.cfg` lists all `dataloop.json` paths (both API and downloadable) for batch publishing
- All category adapters (`llm/base.py`, `vlm/base.py`, etc.) inherit from `NIMBaseAdapter` in `models/api/base_adapter.py`
- E2E tests in `tests/e2e_tests/api/` must mirror the model path structure under `models/api/`
- Test datasets in `tests/assets/e2e_tests/datasets/` are referenced by name in `config.yaml`

## Available Datasets (for tests)

| Name | Contents | Used by |
|---|---|---|
| `text_prompt_text_answer` | Text prompt JSON | LLM, Embeddings |
| `text_image_prompt_text_answer` | Image + text prompt JSON | VLM |
| `chart_image` | Chart image + ontology | Object Detection |
| `deplot_image_prompt_text_answer` | Chart image + image-only prompt | Deplot VLM |

## Integration

All API models require the `dl-ngc-api-key` integration, mapped to the `NGC_API_KEY` environment variable.
