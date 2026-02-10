# NVIDIA NIM Agent

Automated agent for discovering NVIDIA NIM models and onboarding them to the Dataloop marketplace.

## Overview

This agent automates the entire workflow of:
1. **Discovering** new NVIDIA NIM models via API
2. **Comparing** them with existing Dataloop marketplace DPKs
3. **Testing** model adapters locally
4. **Generating** DPK manifests
5. **Publishing** and validating as Dataloop apps
6. **Opening PRs** to add successful models to the repository

## Flow Diagram

```
NVIDIA API --> Fetch Models --> Compare with Dataloop --> Find New Models
                                                               |
                                                               v
                        +----------------------------------+
                        |     FOR EACH NEW MODEL           |
                        |  1. Detect Type (LLM/VLM/Embed)  |
                        |  2. Test Adapter Locally         |
                        |  3. Generate Manifest (MCP)      |
                        |  4. Publish & Test App           |
                        +----------------------------------+
                                                               |
                                                               v
                                            Open GitHub PRs (batched by type)
```

## Components

| File | Description |
|------|-------------|
| `nim_agent.py` | Main orchestrator - coordinates the entire flow |
| `tester.py` | Testing operations - type detection, adapter testing, DPK validation |
| `dpk_mcp_handler.py` | DPK manifest generation via MCP tools |
| `github_client.py` | GitHub operations - branches, commits, PRs |
| `scraping.py` | Web scraping utilities for NVIDIA catalog |
| `main.py` | Entry point and example usage |
| `adapters/` | Model adapter implementations (LLM, VLM, Embedding) |

## Detailed Flow

### Step 1: Fetch NVIDIA Models
- Calls `https://integrate.api.nvidia.com/v1/models` (OpenAI-compatible API)
- Returns list of all available NIM models with metadata

### Step 2: Fetch Dataloop DPKs
- Queries Dataloop marketplace for existing NIM DPKs
- Filters by `scope=public` and `attributes.Category=NIM`

### Step 3: Compare & Find New Models
- Normalizes model names for comparison
- Identifies:
  - **To Add**: Models in NVIDIA but not in Dataloop
  - **Deprecated**: DPKs in Dataloop but no longer in NVIDIA
  - **Matched**: Already onboarded models

### Step 4: Onboard Each Model

For each new model, the `onboard_model()` method runs:

#### 4.1 Detect Model Type
- Uses name-based heuristics:
  - `embed` -> embedding
  - `rerank` -> rerank
  - `vision`, `vlm`, `vl` -> vlm
  - Default -> llm

#### 4.2 Test Adapter Locally
- Loads appropriate adapter (LLM/VLM/Embedding)
- Makes test API call to NVIDIA
- Validates response format

#### 4.3 Generate DPK Manifest
- Calls MCP `create_model_manifest` tool with explicit parameters
- No LLM interpretation - direct parameter mapping
- Returns `dataloop.json` manifest

#### 4.4 Publish & Test App
- Publishes DPK to Dataloop
- Installs as app in test project
- Deploys model service
- Runs test prediction with `PromptItem`
- Cleans up resources after test

### Step 5: Open GitHub PRs
- Creates PRs for successful models
- Batched by model type (embedding/vlm/llm)
- Updates config files:
  - `.bumpversion.cfg` - version tracking
  - `.dataloop.cfg` - manifest registry

## Repository Structure (PRs)

PRs follow this folder structure:
```
models/
  embeddings/
    {publisher}/
      {model_name}/
        dataloop.json
  vlm/
    {publisher}/
      {model_name}/
        dataloop.json
  llm/
    {publisher}/
      {model_name}/
        dataloop.json
```

## Usage

### Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from nim_agent import NIMAgent

# Initialize agent
agent = NIMAgent()

# Run full flow (with limit for testing)
agent.run(limit=5, open_pr=True, pr_by_type=True)
```

### Test Single Model

```python
from nim_agent import NIMAgent

agent = NIMAgent()
result = agent.onboard_model("nvidia/llama-3.1-70b-instruct")
print(result)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NGC_API_KEY` | Yes | NVIDIA NGC API key |
| `DATALOOP_TEST_PROJECT` | Yes | Dataloop project ID for testing |
| `GITHUB_TOKEN` | For PRs | GitHub personal access token |
| `GITHUB_REPO` | For PRs | Target repo (default: `dataloop-ai-apps/nim-api-adapter`) |
| `DPK_MCP_PYTHON` | For DPK | Python path for MCP server (default: `python`) |
| `DPK_MCP_SERVER` | For DPK | Path to MCP DPK generator server script |

## Model Type Detection

| Pattern | Type | Example |
|---------|------|---------|
| `embed`, `e5`, `bge` | embedding | `nvidia/nv-embed-v1` |
| `rerank` | rerank | `nvidia/nv-rerankqa-mistral-4b-v3` |
| `vision`, `vlm`, `vl`, `kosmos` | vlm | `meta/llama-3.2-90b-vision-instruct` |
| Default | llm | `meta/llama-3.1-70b-instruct` |

## Output

The agent generates:
- **Report JSON**: Summary with success/failure counts
- **Manifests JSON**: All generated DPK manifests
- **GitHub PRs**: Per model type (embedding, vlm, llm)

## Error Handling

The agent handles:
- **Adapter test failures**: Skips model, logs error
- **Manifest generation failures**: Skips model, logs error
- **App deployment failures**: Cleans up resources, logs error
- **Cleanup order**: Deletes models -> uninstalls apps -> deletes DPK

## Extending

### Adding New Model Types

1. Add adapter in `adapters/` folder
2. Update `ADAPTER_MAPPING` in `dpk_mcp_handler.py`
3. Update detection patterns in `tester.py`
4. Add folder mapping in `github_client.py`
