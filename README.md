# NVIDIA NIM Adapter for Dataloop

Dataloop adapters for NVIDIA NIM models, enabling seamless integration of NVIDIA's AI models into the Dataloop platform.

## Supported Models

This adapter supports **96 NVIDIA NIM models** across multiple categories:

| Category | Models | Run Anywhere |
|----------|--------|--------------|
| LLM | 80 | 20 |
| Embeddings | 8 | 3 |
| VLM | 5 | 3 |
| Object Detection | 3 | 1 |

**[View Full Support Matrix](support_matrix.md)**

## Quick Start

### Prerequisites

- NVIDIA NGC API Key ([Get one here](https://org.ngc.nvidia.com/setup))
- Dataloop account and SDK

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dataloop-ai-apps/nim-api-adapter.git
cd nim-api-adapter
```

2. Set your NGC API key:
```bash
export NGC_API_KEY=your_api_key_here
```

### Using Models via Dataloop Marketplace

The easiest way to use these models is through the Dataloop Marketplace:

1. Go to **Marketplace** in Dataloop
2. Search for "NIM" 
3. Install the desired model adapter
4. Deploy and start using

## Repository Structure

```
nim-adapter/
├── models/
│   ├── api/                    # API-based model adapters
│   │   ├── llm/               # Large Language Models
│   │   ├── vlm/               # Vision-Language Models
│   │   ├── embeddings/        # Embedding Models
│   │   └── object_detection/  # Object Detection Models
│   └── downloadable/          # Self-hosted model adapters
├── agent/                      # Automated onboarding agent
├── support_matrix.md          # Full list of supported models
└── .dataloop.cfg              # Model manifest registry
```

## Model Types

### API Models
Models that run on NVIDIA's cloud infrastructure. Requires NGC API key.

### Downloadable (Run Anywhere)
Models that can be deployed on your own infrastructure. These are marked with a checkmark in the [Support Matrix](support_matrix.md).

## Documentation

- [Support Matrix](support_matrix.md) - Complete list of supported models
- [Agent Documentation](agent/README.md) - Automated model onboarding
- [Downloadable Models](models/downloadable/README.md) - Self-hosted deployment

## License

See [LICENSE](LICENSE) for details.
