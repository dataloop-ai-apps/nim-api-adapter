import dtlpy as dl
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm"))
from base import ModelAdapter as LLMModelAdapter


class ModelAdapter(LLMModelAdapter):
    """
    VLM adapter for NVIDIA NIM vision models using OpenAI-compatible API (images only).

    Inherits LLM adapter logic (call_model, predict, guided_json, streaming)
    and overrides _prepare_messages to keep image_url content parts instead of
    flattening to plain text.

    Supports Llama Vision, Phi Vision, and other models that use the
    standard OpenAI chat completions format with image_url in messages.
    """

    def _prepare_messages(self, messages: list[dict], context: str = None) -> list[dict]:
        """
        Normalize messages for the VLM API: keep only text and image_url content.
        Content is normalized to list format; video_url parts are skipped (images only).
        """
        normalized = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append(part)
                    elif part.get("type") == "image_url":
                        parts.append(part)
                normalized.append({"role": role, "content": parts if parts else [{"type": "text", "text": ""}]})
            else:
                normalized.append({"role": role, "content": [{"type": "text", "text": content or ""}]})
        return normalized


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    item = dl.items.get(item_id="ITEM_ID_HERE")
    
    adapter = ModelAdapter(model)
    adapter.predict_items([item])
