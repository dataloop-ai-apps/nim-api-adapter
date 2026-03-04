# E2E Test Rules

## Directory Convention

Tests mirror the model directory layout exactly:
```
tests/e2e_tests/api/<category>/<vendor>/<model_folder>/test_<action>/
├── config.yaml      # Test resources and pipeline variables
└── template.json    # Pipeline template (shared per category+action)
```

- **Folder name** = model folder name from `models/api/` (underscores, e.g., `llama_3_1_8b_instruct`)
- **test_predict/** for LLM, VLM, and Object Detection models
- **test_embed/** for Embedding models

## config.yaml Pattern

All config.yaml files follow this structure. The DPK name, model name, and source_app are always identical and come from the model's `dataloop.json` top-level `"name"` field.

```yaml
dpks:
    - name: {dpk_name}            # from dataloop.json "name" field
      install_app: True
      integrations:
        - key: "dl-ngc-api-key"
          value: "NGC_API_KEY"

datasets:
    - name: {dataset_name}        # depends on category (see table below)
      type: local

models:
    - name: {dpk_name}            # same as dpk name
      deploy_model: False
      source_app: {dpk_name}      # same as dpk name

variables:
    - name: dataset
      resource_type: datasets
      resource_value:
          resource_name: {dataset_name}
    - name: model
      resource_type: models
      resource_value:
          resource_name: {dpk_name}
```

## Dataset per Category

| Category | Dataset Name | Template |
|---|---|---|
| LLM | `text_prompt_text_answer` | LLM predict template |
| VLM | `text_image_prompt_text_answer` | VLM predict template |
| Embeddings | `text_prompt_text_answer` | Embed template |
| Object Detection | `chart_image` | Object detection predict template |

Dataset assets live in `tests/assets/e2e_tests/datasets/<dataset_name>/`.

## template.json

Each category has a shared template. When creating new tests, copy from an existing test of the same category:

- **LLM predict**: `tests/e2e_tests/api/llm/meta/llama_3_8b_instruct/test_predict/template.json`
- **VLM predict**: `tests/e2e_tests/api/vlm/meta/llama_3_2_11b_vision_instruct/test_predict/template.json`
- **Embed**: `tests/e2e_tests/api/embeddings/nvidia/nv_embedqa_e5_v5/test_embed/template.json`
- **Object Detection predict**: `tests/e2e_tests/api/object_detection/nv_yolox_page_elements_v1/test_predict/template.json`

## How to Add a New Test

1. Read the model's `dataloop.json` to get the DPK name
2. Determine the category (llm, vlm, embeddings, object_detection)
3. Create `tests/e2e_tests/api/<category>/<vendor>/<model_folder>/test_<action>/`
4. Write `config.yaml` using the pattern above with the correct DPK name and dataset
5. Copy `template.json` from the reference template for that category
