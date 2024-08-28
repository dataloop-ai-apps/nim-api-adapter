from openai import OpenAI, NOT_GIVEN
import dtlpy as dl
import requests
import logging
import json
import os

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key: NGC_API_KEY")
        self.api_key = os.environ.get("NGC_API_KEY", None)
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
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def stream_response(self, messages):
        if self.nim_model_name.startswith('vlm/'):
            response = self.call_model_requests(messages=messages)
        else:
            response = self.call_model_open_ai(messages=messages)

        if self.stream:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        else:
            yield response

    @staticmethod
    def process_instruct_messages(messages):
        for message in messages:
            if message.get('role') == 'user':
                return [message]

    def call_model_open_ai(self, messages):
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        current_dir = os.path.dirname(__file__)
        nim_configs_filepath = os.path.join(current_dir, 'nim_configs.json')
        with open(nim_configs_filepath, 'r') as f:
            model_configs = json.load(f)

        model_type = model_configs.get(self.nim_model_name, None)
        if model_type == 'instruct':
            messages = self.process_instruct_messages(messages)
            full_answer = self.call_chat_model(messages=messages, client=client)
        elif model_type == 'chat':
            full_answer = self.call_chat_model(messages=messages, client=client)
        elif model_type == 'reward':
            full_answer = self.call_reward_model(messages=messages, client=client)
        elif model_type == 'coding':
            full_answer = self.call_coding_model(messages=messages, client=client)
        else:
            raise ValueError(f"Model type {model_type} for model {self.nim_model_name} is not supported")

        return full_answer

    def call_reward_model(self, messages, client):
        self.stream = False

        reward_dict = dict()
        if messages[-1].get('role') == 'assistant':
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages
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
        if self.stream is False:
            full_answer = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    full_answer += chunk.choices[0].delta.content
        else:
            full_answer = completion
        return full_answer

    def call_coding_model(self, messages, client):
        self.stream = False

        messages = self.process_instruct_messages(messages=messages)
        code = client.completions.create(
            model=self.nim_model_name,
            prompt=messages[0].get('content', ''),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            stream=self.stream
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

        self.stream = False
        full_answer = response.json().get('choices')[0].get('message').get('content')
        return full_answer

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        for prompt_item in batch:
            # Get all messages including model annotations
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            messages.insert(0, {"role": "system", "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(
                    nearest_items=nearest_items,
                    add_metadata=self.configuration.get("add_metadata")
                )
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            stream_response = self.stream_response(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
                    stream=True,
                    model_info={
                        'name': self.model_entity.name,
                        'confidence': 1.0,
                        'model_id': self.model_entity.id
                    }
                )

            return []


if __name__ == '__main__':
    print(os.path.dirname(__file__))
    # model = dl.models.get(model_id='')
    # item = dl.items.get(item_id='')
    # adapter = ModelAdapter(model)
    # adapter.predict_items(items=[item])
