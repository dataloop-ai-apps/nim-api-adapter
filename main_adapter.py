from openai import OpenAI
import dtlpy as dl
import requests
import logging
import json
import os

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        if os.environ.get("NVIDIA_NIM_API_KEY", None) is None:
            raise ValueError(f"Missing API key: NVIDIA_NIM_API_KEY")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

    def prepare_item_func(self, item: dl.Item):
        if ('json' not in item.mimetype or
                item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
            raise ValueError('Only prompt items are supported')
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def call_model_open_ai(self, messages):
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_NIM_API_KEY", None)
        )
        completion = client.chat.completions.create(
            model=self.nim_model_name,
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
        return full_answer

    def call_model_requests(self, messages):
        url = f"https://ai.api.nvidia.com/v1/{self.model_entity.configuration.get('nim_model_name')}"
        headers = {
            "Authorization": f"Bearer {os.environ.get('NVIDIA_NIM_API_KEY', None)}",
            "Accept": "application/json"
        }
        max_token = self.model_entity.configuration.get('max_token', 1024)
        temperature = self.model_entity.configuration.get('temperature', 0.2)
        top_p = self.model_entity.configuration.get('top_p', 0.7)
        seed = self.model_entity.configuration.get('seed', 0)

        payload = {
            "messages": messages,
            "max_tokens": max_token,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False
        }
        response = requests.post(url=url, headers=headers, json=payload)
        if not response.ok:
            raise ValueError(f'error:{response.status_code}, message: {response.text}')
        full_answer = response.json().get('choices')[0].get('message').get('content')
        return full_answer

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', '')

        annotations = []
        for prompt_item in batch:
            ann_collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                # get latest question
                question = [p['value'] for p in prompt_content if 'text' in p['mimetype']][0]
                nearest_items = [p['nearestItems'] for p in prompt_content if 'metadata' in p['mimetype']]
                if len(nearest_items) > 0:
                    nearest_items = nearest_items[0]
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
                if self.nim_model_name.startswith('vlm/'):
                    full_answer = self.call_model_requests(messages=messages)
                else:
                    full_answer = self.call_model_open_ai(messages=messages)
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
