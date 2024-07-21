from openai import OpenAI, NOT_GIVEN
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
        self.api_key = os.environ.get("NVIDIA_NIM_API_KEY", None)
        self.max_token = model_entity.configuration.get('max_token', 1024)
        self.temperature = model_entity.configuration.get('temperature', 0.2)
        self.top_p = model_entity.configuration.get('top_p', 0.7)
        self.seed = model_entity.configuration.get('seed', 0)
        self.stream = model_entity.configuration.get('stream', True)
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
            api_key=self.api_key
        )
        model_type = self.model_entity.metadata.get('model_type', 'chat')
        if model_type == 'chat':
            full_answer = self.call_chat_model(messages=messages, client=client)
        elif model_type == 'reward':
            full_answer = self.call_reward_model(messages=messages, client=client)
        else:
            full_answer = self.call_coding_model(messages=messages, client=client)

        return full_answer

    def call_reward_model(self, messages, client):
        reward_dict = dict()
        if messages[-1].get('role') == 'assistant':
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages,
            )
            rewards = completion.choices[0].logprobs.content
            for reward in rewards:
                reward_dict[reward.token] = reward.logprob
        return str(reward_dict)

    def call_chat_model(self, messages, client):
        completion = client.chat.completions.create(
            model=self.nim_model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            stream=self.stream
        )
        full_answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_answer += chunk.choices[0].delta.content
        return full_answer

    def call_coding_model(self, messages, client):
        code = client.completions.create(
            model=self.nim_model_name,
            prompt=messages[0].get('content', ''),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            stream=False
        )
        code = code.choices[0].text
        return code

    def call_model_requests(self, messages):
        url = f"https://ai.api.nvidia.com/v1/{self.model_entity.configuration.get('nim_model_name')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "messages": messages,
            "max_tokens": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": self.stream
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
                messages = list()
                if len(system_prompt) > 0:
                    messages.append({"role": "system",
                                     "content": system_prompt})
                messages.extend([
                    {"role": "user",
                     "content": question + ' ' + context}
                ])
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
