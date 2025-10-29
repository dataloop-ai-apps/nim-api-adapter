NV-CLIP NIM Embeddings Adapter (OpenAI-compatible)

Overview
This adapter boots one or more NVIDIA NV-CLIP NIM instances inside the container and exposes an OpenAI-compatible embeddings API that the adapter calls locally. It supports multi-instance startup on different ports and random request distribution with retries.

Key options (configuration fields)
- num_servers: Number of NIM instances to start in the container. Default: 1
- nim_base_port: Base HTTP port for the first instance. Each next instance uses base_port + 10*idx. Default: env NIM_HTTP_API_PORT or 8000
- nim_model_name: Model identifier to send in the OpenAI embeddings request. Default: nvidia/nvclip-vit-h-14

Relevant environment variables (passed to each instance)
- NGC_API_KEY (required): Your NGC/NGC API key used by NIM to fetch artifacts
- NVIDIA_API_KEY (optional): Same as NGC_API_KEY if provided
- ACCEPT_EULA: Accept NVIDIA EULA (Y). Default: Y
- NIM_HTTP_API_PORT: Internal service port inside the container for each instance (the adapter sets a distinct value per instance)
- NIM_LOG_LEVEL: Log level for the NIM. Default: warning
- NIM_MANIFEST_PROFILE: Optimized profile to select for your GPU (see docs below)
- NIM_CACHE_ROOT / NIM_CACHE_PATH: Cache directory for model artifacts
- NGC_ORG / NGC_TEAM: Optional org/team scoping for NGC

How multi-instance works
- The adapter launches num_servers processes of the NIM, each with a distinct NIM_HTTP_API_PORT derived from nim_base_port.
- It waits for readiness on /v1/health/ready (fallback /v1/models) for each instance.
- Calls to embed are sent to a random healthy instance; on failure, the adapter retries the remaining instances in random order.
- Per-instance logs are written to /tmp/nim_startup_<port>.log and also streamed to stdout with a [nim:<idx>] prefix.

Usage tips for throughput
- Increase request concurrency and/or batch size to better utilize the GPU.
- Set NIM_MANIFEST_PROFILE to the optimized profile for your GPU. This can significantly improve throughput.
- Prefer one NIM per container for production; if running multiple in one container, ensure distinct ports and watch for possible internal port conflicts.

References
- NVIDIA NV-CLIP NIM configuration: https://docs.nvidia.com/nim/nvclip/latest/configuration.html


