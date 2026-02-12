import dtlpy as dl
import logging
import os
import sys
import json

# Add parent directory to path so we can import the shared base
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_adapter import NIMBaseAdapter, logger


class ModelAdapter(NIMBaseAdapter):

    def prepare_item_func(self, item: dl.Item):
        return dl.PromptItem.from_item(item=item)

    def _flatten_messages(self, messages: list[dict]) -> list[dict]:
        """
        Flatten message content from array format to plain string.
        
        Some NVIDIA NIM LLM models expect plain string content, not the 
        OpenAI multimodal format [{"type": "text", "text": "..."}].
        """
        flattened = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            if isinstance(content, list):
                # Extract text from array format
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif "text" in part:
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                flattened.append({"role": role, "content": " ".join(text_parts)})
            else:
                flattened.append({"role": role, "content": content})
        
        return flattened

    def call_model(self, messages: list[dict]):
        """Call NVIDIA NIM chat completions API."""
        # Flatten messages - LLMs expect plain string content, not multimodal arrays
        messages = self._flatten_messages(messages)
        
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
                    
        # NVIDIA API requires guided_json inside "nvext", not at root
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
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name

        for prompt_item in batch:
            # Get all messages including model annotations
            messages = prompt_item.to_messages(model_name=model_name)
            if system_prompt and system_prompt.strip():
                messages.insert(0, {"role": "system", "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

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

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    print(f"Nim Model name: {model.configuration.get('nim_model_name')}")
    
    item = dl.items.get(item_id="ITEM_ID_HERE")
    adapter = ModelAdapter(model)
    adapter.predict_items([item])
