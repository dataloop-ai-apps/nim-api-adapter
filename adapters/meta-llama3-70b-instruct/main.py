import logging
import json
import os
import dtlpy as dl
from openai import OpenAI

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        api_key = os.environ.get("NVIDIA_NIM_API_KEY", None)
        if api_key is None:
            raise ValueError(f"Missing API key: NVIDIA_NIM_API_KEY")
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
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
                    model="meta/llama3-70b-instruct",
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
    if os.path.isfile('.env'):
        with open('.env') as f:
            # os.environ['NVIDIA_NIM_API_KEY'] = json.load(f)["NVIDIA_NIM_API_KEY"]
            os.environ['NVIDIA_NIM_API_KEY'] = "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"
    model = dl.models.get(model_id='66564cc366683320adc4b8ee')
    item = dl.items.get(item_id='66564da2ae188546873cd72a')
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
