import dtlpy as dl
import logging
import os
import sys

# Add parent directory to path so we can import the shared base
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_adapter import NIMBaseAdapter, logger


class ModelAdapter(NIMBaseAdapter):

    def call_model_open_ai(self, text):
        kwargs = dict(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
        )
        if self.use_nvidia_extra_body:
            kwargs["extra_body"] = {"input_type": "query", "truncate": "NONE"}
        try:
            response = self.client.embeddings.create(**kwargs)
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Embeddings API call failed. Base URL: {self.base_url}, Model: {self.nim_model_name}, Error: {e}")
            raise

    def embed(self, batch, **kwargs):
        
        if self.using_downloadable:
            self.check_jwt_expiration()
            
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                text = item
            else:
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][0]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][0]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            embedding = self.call_model_open_ai(text)
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    print(f"Nim Model name: {model.configuration.get('nim_model_name')}")
    
    item = dl.items.get(item_id="ITEM_ID_HERE")
    adapter = ModelAdapter(model)
    result = adapter.embed_items([item])
    
    print(f"Embedding dimension: {len(result[0]) if result else 'N/A'}")
