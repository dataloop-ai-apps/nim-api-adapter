# Downloadable NIM Models for Dataloop

Deploy NVIDIA NIM models as downloadable services on Dataloop platform.

## Overview

This project provides templates and scripts to:
1. Build Docker images that wrap NVIDIA NIM models with Dataloop agent support
2. Deploy these images as Dataloop services with OpenAI-compatible API endpoints

## Quick Start

### 1. Build the Docker Image

```bash
bash build_downloadable_nim.sh nvclip
```

This builds and pushes `gcr.io/viewo-g/piper/agent/runner/gpu/nvclip:1.0.0`

### 2. Deploy to Dataloop

```bash
python deploy.py --model nvclip --project "Your Project Name"
```

Options:
- `--build` / `-b`: Build Docker image before deploying
- `--clean` / `-c`: Clean existing installations first
- `--version` / `-v`: Docker image version (default: 0.1.13)
- `--env` / `-e`: Dataloop environment (prod/dev/rc)

### 3. Test the Deployment

```bash
python test_deployment.py --app-id <your-app-id>
```

## Project Structure

```
.
├── main.py                 # Service runner - starts NIM server and streams logs
├── Dockerfile.template     # Docker template for building NIM images
├── manifest_template.json  # Dataloop manifest template
├── deploy.py              # Deployment script
├── build_downloadable_nim.sh  # Docker build script
├── test_deployment.py     # Test script for deployed services
├── dataloop.json          # Current dataloop manifest
└── nv-clip-downloadable/  # Working example for nvclip
```

## How It Works

### Docker Image

The Dockerfile.template:
1. Extends the official NVIDIA NIM image (`nvcr.io/nim/nvidia/<model>:latest`)
2. Installs Dataloop SDK and agent packages
3. Configures the server to run on port 3000

### Service Runner

The `main.py` runner:
1. Starts the NIM server (`/opt/nim/start_server.sh`)
2. Streams stdout/stderr for real-time logging
3. Exposes an OpenAI-compatible API

### API Endpoints

Once deployed, the service exposes:
- `/v1/embeddings` - Generate embeddings (for embedding models like nvclip)
- `/v1/completions` - Generate completions (for LLM models)
- `/docs` - OpenAPI documentation
- `/v1/metadata` - Model metadata

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NGC_API_KEY` | (required) | NVIDIA NGC API key |
| `NIM_CACHE_PATH` | `/tmp/.nim` | Model cache path |
| `NIM_HTTP_API_PORT` | `3000` | Server port |

### Runtime Settings

In `manifest_template.json`:
- `podType`: GPU type (e.g., `gpu-t4-m`)
- `numReplicas`: Number of replicas
- `concurrency`: Concurrent requests per replica

## Supported Models

Any NVIDIA NIM model from `nvcr.io/nim/nvidia/`:
- `nvclip` - CLIP-based embeddings
- `phi-3-mini-4k-instruct` - Phi-3 LLM
- And more...

## Testing Your Deployment

### Via Swagger UI (Recommended)

The easiest way to test is via the built-in Swagger UI:

```
https://<service-name>-<app-id>.apps.dataloop.ai/docs
```

Example: `https://nvclip-downloadable-runner-69885809eb577aee86877518.apps.dataloop.ai/docs`

### Via Python Script

```bash
python test_deployment.py --app-id <your-app-id>
```

The script uses the Dataloop SDK's `gen_request` for authentication.

### Via SDK (Programmatic Access)

```python
import dtlpy as dl

dl.setenv('prod')
dl.login()

app_id = '<your-app-id>'
app = dl.apps.get(app_id=app_id)

# Get the panel route
panel_name = list(app.routes.keys())[0]
gate_path = app.routes[panel_name].rstrip('/')

# Make request using SDK auth
data = {
    "input": ["Hello world"],
    "model": "nvidia/nvclip-vit-h-14",
    "encoding_format": "float"
}

success, response = dl.client_api.gen_request(
    req_type='post',
    path=f"{gate_path}/v1/embeddings",
    json_req=data
)

if response.status_code == 200:
    print(response.json())
```

**Note**: The `.apps.dataloop.ai` domain requires browser-based cookie authentication.
For programmatic access, use the SDK's `gen_request` with the gate panel route, or test 
via the Swagger UI.
