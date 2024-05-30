import logging
import json
import os
import dtlpy as dl
from openai import OpenAI

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model, nvidia_api_key_name=None):
        self.api_key = os.environ.get(nvidia_api_key_name, None)
        if self.api_key is None:
            raise ValueError(f"Missing API key: {nvidia_api_key_name}")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

    def prepare_item_func(self, item: dl.Item):
        if ('json' not in item.mimetype or
                item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
            raise ValueError('Only prompt items are supported')
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def predict(self, batch, **kwargs):
        annotations = []
        for prompt_item in batch:
            ann_collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                text = None
                for partial_prompt in prompt_content:
                    if 'text' in partial_prompt.get('mimetype', ''):
                        text = partial_prompt.get('value')
                    else:
                        logger.warning(f"Prompt is missing text prompt.")
                if text is None:
                    logger.warning(f"{prompt_name} is missing text prompt.")
                    continue
                messages = [{"role": "user",
                             "content": text}]
                model_name = "ibm/granite-34b-code-instruct"
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.5,
                    top_p=1,
                    max_tokens=1024,
                    stream=True
                )
                full_answer = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        full_answer += chunk.choices[0].delta.content
                ann_collection.add(
                    annotation_definition=dl.FreeText(text=full_answer),
                    prompt_id=prompt_name,
                    model_info={
                        'name': self.model_entity.name,
                        'model_id': self.model_entity.id,
                        'confidence': 1.0
                    }
                )
            annotations.append(ann_collection)
        return annotations


if __name__ == '__main__':
    model = dl.models.get(model_id='65dd0f08ce79b0cf60e95074')
    item = dl.items.get(item_id='66159ece6e626c8430542f32')
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
