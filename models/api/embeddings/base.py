from openai import OpenAI
import dtlpy as dl
import logging
import os

# Toggleable logger - set NIM_DISABLE_LOGGING=1 to disable
if os.environ.get("NIM_DISABLE_LOGGING", "").lower() in ("1", "true", "yes"):
    logger = logging.getLogger("NIM Adapter")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
else:
    logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        
        self.base_url = self.configuration.get("base_url", "https://integrate.api.nvidia.com/v1")
        logger.info(f"Using base URL: {self.base_url}")
        
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")
        
        self.api_key = os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing NGC_API_KEY environment variable")
        
        # Create OpenAI client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Validate API key early (no token consumption)
        # Calls /v1/models to verify auth; if invalid, fails fast before any inference
        if self.base_url.rstrip("/") == "https://integrate.api.nvidia.com/v1":
            try:
                self.client.models.list()
                logger.info(f"API key validated for {self.nim_model_name}, base URL: {self.base_url}")
            except Exception as e:
                raise ValueError(f"API key validation failed: {e}")
        else:
            logger.info(f"Skipping API key validation for {self.nim_model_name}, base URL: {self.base_url}")
        

    def call_model_open_ai(self, text):
        response = self.client.embeddings.create(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
            extra_body={
                "input_type": "query",
                "truncate": "NONE"
            }
        )
        embedding = response.data[0].embedding
        return embedding

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
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