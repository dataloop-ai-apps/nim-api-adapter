from openai import OpenAI
import dtlpy as dl
import logging
import os

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key: NGC_API_KEY")
        self.api_key = os.environ.get("NGC_API_KEY", None)
        self.embedding_size = model_entity.configuration.get('embeddings_size', 512)
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

    def call_model_open_ai(self, text):
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        response = client.embeddings.create(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
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
                            text = messages['content']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            embedding = self.call_model_open_ai(text)
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings


if __name__ == '__main__':
    dl.setenv('rc')
    import dotenv

    dotenv.load_dotenv()
    model = dl.models.get(model_id='')
    item = dl.items.get(item_id='')
    adapter = ModelAdapter(model)
    adapter.embed_items(items=[item])
