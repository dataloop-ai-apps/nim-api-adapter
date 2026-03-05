# NVIDIA NIM Adapter for Dataloop

Dataloop model adapters for [NVIDIA NIM](https://build.nvidia.com/) — providing access to NVIDIA's inference models through the Dataloop platform.

Models are available in two deployment modes: **API** (hosted by NVIDIA) and **Downloadable** (self-hosted on Dataloop infrastructure). Both modes expose an OpenAI-compatible interface and share the same adapter codebase.

## Deployment Modes

### API Models

API models run on NVIDIA's hosted infrastructure. Inference requests are sent to `https://integrate.api.nvidia.com/v1` using an NGC API key. No GPU resources are required on your side.

**How it works:**
1. Install the model DPK from the Dataloop marketplace
2. Provide your NGC API key via the `dl-ngc-api-key` integration
3. The adapter forwards requests to NVIDIA's API and returns results

**Configuration:**

| Field | Description |
|-------|-------------|
| `nim_model_name` | NVIDIA model identifier (e.g. `meta/llama-3.1-8b-instruct`) |
| `base_url` | NVIDIA API endpoint (default: `https://integrate.api.nvidia.com/v1`) |
| `max_tokens` | Maximum tokens to generate (LLM/VLM) |
| `temperature` | Sampling temperature (LLM/VLM) |
| `stream` | Enable streaming responses (LLM/VLM) |
| `system_prompt` | System prompt (LLM/VLM) |
| `embeddings_size` | Embedding vector dimension (Embeddings) |

### Downloadable Models

Downloadable models are self-hosted: they run inside Dataloop-managed GPU services using official NVIDIA NIM container images from `nvcr.io`. Instead of calling NVIDIA's cloud API, the adapter talks to a local NIM server running on the same infrastructure.

**How it works:**

1. A Docker image is built from the official NVIDIA NIM container (`nvcr.io/nim/<model>:latest`), extended with Dataloop SDK and agent packages
2. When installed, Dataloop deploys this image as a GPU-backed service that runs the NIM inference server (`start_server.sh`)
3. The service exposes an OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`) accessible within the Dataloop network
4. The model adapter connects to this service instead of NVIDIA's cloud — the adapter resolves the service URL from the app installation and authenticates

**Connecting a model to a downloadable service:**

When you install a downloadable NIM app, it creates a running service with its own app ID. To use it with a model adapter:

1. Install the downloadable app from the marketplace (this provisions the GPU service and starts the NIM server)
2. Copy the app's installation ID (`app_id`)
3. Set `app_id` in the model's configuration — this tells the adapter to route inference to the downloadable service instead of NVIDIA's cloud API

```json
{
  "nim_model_name": "meta/llama-3.1-8b-instruct",
  "app_id": "<your-downloadable-app-installation-id>"
}
```

When `app_id` is present, the adapter resolves the service endpoint via Dataloop's app routing, obtains a JWT session cookie, and creates an OpenAI client pointed at the local service. JWT sessions are automatically refreshed before expiration.

## Model Types

| Type | Adapter | Description |
|------|---------|-------------|
| LLM | `models/api/llm/base.py` | Chat completion models (Llama, Mistral, etc.) |
| VLM | `models/api/vlm/base.py` | Vision-language models with image understanding |
| Embeddings | `models/api/embeddings/base.py` | Text embedding models |
| Object Detection | `models/api/object_detection/base.py` | Vision models with bounding box output |

All adapters inherit from `NIMBaseAdapter` (`models/api/base_adapter.py`), which handles client setup, downloadable endpoint resolution, and JWT session management.

## Repository Structure

```
models/
  api/                          # API model adapters and manifests
    base_adapter.py             # Shared base class (client setup, JWT, health check)
    llm/
      base.py                   # LLM adapter
      {publisher}/{model}/dataloop.json
    vlm/
      base.py                   # VLM adapter (inherits from LLM)
      {publisher}/{model}/dataloop.json
    embeddings/
      base.py                   # Embeddings adapter
      {publisher}/{model}/dataloop.json
    object_detection/
      base.py                   # Object detection adapter
      {publisher}/{model}/dataloop.json
  downloadable/                 # Downloadable NIM services
    main.py                     # Shared service runner (starts NIM server, streams logs)
    {type}/{publisher}/{model}/dataloop.json
agent/                          # Automated model onboarding agent
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NGC_API_KEY` | Yes | NVIDIA NGC API key for model access |

## Requirements

- Python 3.10+
- `dtlpy` (Dataloop SDK)
- `openai` (OpenAI Python client)
- `httpx`, `PyJWT` (for downloadable auth)
