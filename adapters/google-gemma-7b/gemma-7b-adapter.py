import logging
import json
import os
import dtlpy as dl
from openai import OpenAI

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model, nvidia_api_key_name):
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
        system_prompt = self.model_entity.configuration.get('system_prompt', "")

        annotations = []
        for prompt_item in batch:
            ann_collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                # get latest question
                question = [p['value'] for p in prompt_content if 'text' in p['mimetype']][0]
                nearest_items = [p['nearestItems'] for p in prompt_content if 'metadata' in p['mimetype']][0]
                # build context
                context = ""
                for item_id in nearest_items:
                    context_item = dl.items.get(item_id=item_id)
                    with open(context_item.download(), 'r', encoding='utf-8') as f:
                        text = f.read()
                    context += f"\n{text}"
                messages = [{"role": "system",
                             "content": system_prompt},
                            {"role": "assistant",
                             "content": context},
                            {"role": "user",
                             "content": question}]
                completion = self.client.chat.completions.create(
                    model="google/gemma-7b",
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
    model = dl.models.get(model_id='')
    item = dl.items.get(item_id='')
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
