# DPK Generator MCP Client
"""
Client for generating DPK manifests for NVIDIA NIM models.

Calls MCP tools directly with explicit parameters (no LLM interpretation).
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# MCP server config - set via environment variables
# PYTHON_PATH: Path to Python executable for MCP server
# MCP_SERVER_PATH: Path to MCP server script
PYTHON_PATH = os.environ.get("PYTHON_PATH", "python")
MCP_SERVER_PATH = os.environ.get("MCP_SERVER_PATH")

# Adapter paths mapping - relative to repo root (models/api/ folder)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_MAPPING = {
    "embedding": "models/api/embeddings/base.py",
    "vlm": "models/api/vlm/base.py",
    "llm": "models/api/llm/base.py"
}

# DPK version - keep in sync with codebase
DPK_VERSION = "0.3.34"

# Model type to DPK category/type mapping
# Includes attributes for the manifest
MODEL_TYPE_CONFIG = {
    "llm": {
        "model_category": "Gen AI",
        "model_type": "LLM",
        "input_type": "text",
        "output_type": "text",
        # Attributes
        "media_type": ["Text"],
        "gen_ai": "LLM",
        "nlp": "Conversational"
    },
    "vlm": {
        "model_category": "Gen AI",
        "model_type": "LMM",
        "input_type": "image",
        "output_type": "text",
        # Attributes
        "media_type": ["Multi Modal"],
        "gen_ai": "LMM",
        "nlp": "Conversational"
    },
    "embedding": {
        "model_category": "NLP",
        "model_type": "Embeddings",
        "input_type": "text",
        "output_type": "embedding",
        # Attributes (no Gen AI for embeddings)
        "media_type": ["Text"],
        "nlp": "Embeddings"
    }
}


class DPKGeneratorClient:
    """
    MCP Client for generating DPK manifests for NVIDIA NIM models.
    
    Calls create_model_manifest MCP tool directly with explicit parameters.
    
    Requires environment variables:
    - DPK_MCP_PYTHON: Path to Python executable (default: "python")
    - DPK_MCP_SERVER: Path to MCP server script (required)
    """
    
    def __init__(self):
        if not MCP_SERVER_PATH:
            raise ValueError(
                "DPK_MCP_SERVER environment variable required. "
                "Set it to the path of the MCP server script."
            )
        
        self.server_params = StdioServerParameters(
            command=PYTHON_PATH,
            args=[MCP_SERVER_PATH],
            env={
                **os.environ,
                "NGC_API_KEY": os.environ.get("NGC_API_KEY", ""),
            }
        )
    
    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the result."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return json.loads(result.content[0].text)
    
    def create_nim_dpk_manifest(self, model_id: str, model_type: str) -> dict:
        """
        Create a DPK manifest for a NVIDIA NIM model.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            model_type: Type of model ("llm", "vlm", "embedding")
            
        Returns:
            dict with status, dpk_name, manifest, adapter_path, error
        """
        result = {
            "status": "pending",
            "dpk_name": None,
            "manifest": None,
            "adapter_path": None,
            "adapter_code": None,
            "error": None
        }
        
        dpk_name = self.model_to_dpk_name(model_id)
        result["dpk_name"] = dpk_name
        
        try:
            # Get adapter path
            adapter_rel_path = ADAPTER_MAPPING.get(model_type, "models/api/llm/base.py")
            adapter_path = os.path.join(REPO_ROOT, adapter_rel_path)
            result["adapter_path"] = adapter_path
            
            # Read adapter code
            with open(adapter_path, 'r') as f:
                result["adapter_code"] = f.read()
            
            # Get type config
            type_config = MODEL_TYPE_CONFIG.get(model_type, MODEL_TYPE_CONFIG["llm"])
            
            # Build display name from model_id
            display_name = model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()
            
            # Extract provider from model_id (e.g., "meta/llama" → "Meta")
            model_provider = self._get_model_provider(model_id)
            
            # Build attributes based on model type
            attributes = {
                "Hub": ["Nvidia", "Dataloop"],
                "Provider": model_provider,
                "Deployed By": "NVIDIA",
                "Category": ["Model", "NIM"],
                "Media Type": type_config["media_type"],
                "NLP": type_config["nlp"]
            }
            # Add "Gen AI" only for LLM/VLM (not for embeddings)
            if "gen_ai" in type_config:
                attributes["Gen AI"] = type_config["gen_ai"]
            
            # Call MCP tool directly with explicit parameters (no LLM)
            manifest = asyncio.run(self._call_tool("create_model_manifest", {
                "name": dpk_name,
                "display_name": f"NIM {display_name}",
                "description": f"NVIDIA NIM adapter for {model_id}",
                "version": DPK_VERSION,
                "scope": "public",  # Final manifest is public; testing overrides to "project"
                "model_category": type_config["model_category"],
                "model_type": type_config["model_type"],
                "provider": model_provider,
                "trainable": False,
                "input_type": type_config["input_type"],
                "output_type": type_config["output_type"],
                "entry_point": adapter_rel_path,
                "class_name": "ModelAdapter",
                "runner_image": "gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.14",
                "integrations": ["dl-ngc-api-key"],  # NGC API key secret
                "hub": ["Nvidia", "Dataloop"],
                "git_url": "https://github.com/dataloop-ai-apps/nim-api-adapter",
                "git_tag": DPK_VERSION,
                "configuration": {
                    "nim_model_name": model_id,
                    "base_url": "https://integrate.api.nvidia.com/v1"
                },
                "attributes": attributes
            }))
            
            result.update({
                "status": "success",
                "manifest": manifest
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result.update({
                "status": "error",
                "error": str(e)
            })
        
        return result
    
    @staticmethod
    def model_to_dpk_name(model_id: str) -> str:
        """Convert NVIDIA model ID to DPK name."""
        name = model_id.split("/")[-1]
        name = name.replace(".", "-").replace("_", "-").lower()
        return f"nim-{name}"
    
    @staticmethod
    def _get_model_provider(model_id: str) -> str:
        """
        Extract and format provider name from model_id.
        
        Examples:
            "meta/llama-3.1-8b" → "Meta"
            "nvidia/nv-embed-v1" → "NVIDIA"
            "deepseek-ai/deepseek-r1" → "DeepSeek"
            "mistralai/mistral-7b" → "Mistral AI"
            "google/gemma-7b" → "Google"
            "microsoft/phi-3" → "Microsoft"
            "ibm/granite-34b" → "IBM"
        """
        # Known provider mappings
        provider_map = {
            "meta": "Meta",
            "nvidia": "NVIDIA",
            "deepseek-ai": "DeepSeek",
            "deepseek_ai": "DeepSeek",
            "mistralai": "Mistral AI",
            "google": "Google",
            "microsoft": "Microsoft",
            "ibm": "IBM",
            "bigcode": "BigCode",
            "snowflake": "Snowflake",
            "baidu": "Baidu",
            "openai": "OpenAI",
        }
        
        # Extract provider from model_id
        if "/" in model_id:
            provider_raw = model_id.split("/")[0].lower()
        else:
            provider_raw = "nvidia"  # Default
        
        # Return mapped name or title-cased version
        return provider_map.get(provider_raw, provider_raw.replace("-", " ").replace("_", " ").title())
    
    @staticmethod
    def get_adapter_path(model_type: str) -> str:
        """Get the adapter file path for a model type."""
        adapter_rel_path = ADAPTER_MAPPING.get(model_type, "models/api/llm/base.py")
        return os.path.join(REPO_ROOT, adapter_rel_path)


# =========================================================================
# Module-level convenience functions
# =========================================================================

def create_nim_manifest(model_id: str, model_type: str) -> dict:
    """Create NIM DPK manifest (module-level convenience function)."""
    client = DPKGeneratorClient()
    return client.create_nim_dpk_manifest(model_id, model_type)


# Usage examples
if __name__ == "__main__":
    # Test LLM manifest
    print("=== LLM Manifest ===")
    result = create_nim_manifest("meta/llama-3.1-8b-instruct", "llm")
    print(f"Status: {result['status']}")
    print(f"DPK Name: {result['dpk_name']}")
    if result['manifest']:
        print(json.dumps(result['manifest'], indent=2))
    
    # Test Embedding manifest
    print("\n=== Embedding Manifest ===")
    result = create_nim_manifest("nvidia/nv-embed-v1", "embedding")
    print(f"Status: {result['status']}")
    print(f"DPK Name: {result['dpk_name']}")
    if result['manifest']:
        print(json.dumps(result['manifest'], indent=2))
