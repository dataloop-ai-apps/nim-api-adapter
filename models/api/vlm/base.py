import dtlpy as dl
import logging
import os
import sys
import json

# Add parent directory to path so we can import the shared base
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_adapter import NIMBaseAdapter, logger


class ModelAdapter(NIMBaseAdapter):
    """
    VLM adapter for NVIDIA NIM vision models using OpenAI-compatible API (images only).

    Supports Llama Vision, Phi Vision, and other models that use the
    standard OpenAI chat completions format with image_url in messages.
    Guided JSON output (guided_json in configuration) is sent via nvext like the LLM adapter.
    """

    def prepare_item_func(self, item: dl.Item):
        return dl.PromptItem.from_item(item=item)
    
    def call_model(self, messages: list[dict]):
        """Call NVIDIA NIM chat completions API (images only)."""
        messages = self._prepare_image_content(messages)
    
        stream = self.configuration.get("stream")
        max_tokens = self.configuration.get("max_tokens", 512)
        temperature = self.configuration.get("temperature", 0.2)
        top_p = self.configuration.get("top_p", 0.7)
        # Schema in model config only (inline JSON or dict)
        guided_json = self.configuration.get("guided_json", None)
        if guided_json is not None:
            try:
                guided_json = json.loads(guided_json) if isinstance(guided_json, str) else guided_json
            except Exception as e:
                logger.error(f"Error parsing guided_json: {e}")
                guided_json = None
        
        extra_body = {}
        if guided_json and self.use_nvidia_extra_body:
            extra_body["nvext"] = {"guided_json": guided_json}
            logger.info(f"Using guided_json in nvext: {guided_json}")

        # Build kwargs - omit seed as some models reject it
        kwargs = {
            "model": self.nim_model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**kwargs)

        if stream is True:
            for chunk in response:
                if not chunk.choices:
                    continue
                yield chunk.choices[0].delta.content or ""
        else:
            yield response.choices[0].message.content or ""

    def predict(self, batch, **kwargs):
        """Run prediction on a batch of prompts."""
        if self.using_downloadable:
            self.check_jwt_expiration()

        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        model_name = self.model_entity.name
        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            
            if system_prompt and system_prompt.strip():
                messages.insert(0, {"role": "system", "content": system_prompt})

            stream_response = self.call_model(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                model_info={'name': model_name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []

    def _prepare_image_content(self, messages: list[dict]) -> list[dict]:
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
