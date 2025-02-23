from openai import OpenAI
import dtlpy as dl
import requests
import logging
import json
import os

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key")

        self.adapter_defaults.upload_annotations = False

        self.api_key = os.environ.get("NGC_API_KEY", None)
        self.max_token = self.configuration.get('max_token', 1024)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.7)
        self.seed = self.configuration.get('seed', 0)
        self.stream = self.configuration.get('stream', True)
        self.json_schema = self.configuration.get('json_schema', None)

        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")
        self.nim_invoke_url = self.configuration.get("nim_invoke_url", self.nim_model_name)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    @staticmethod
    def process_chat_messages(messages):
        reformatted_messages = list()
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user' or role == 'assistant':
                if not isinstance(content, str):
                    msg = {'role': role, 'content': content[0].get(content[0].get("type", "text"), "")}

            reformatted_messages.append(msg)

        return reformatted_messages

    @staticmethod
    def process_multimodal_messages(messages: list):
        reformatted_messages = []

        for message in messages:
            if message.get('role') == 'system':
                continue
            # Extract image base64 string, find first, default if not match
            image_url = next(
                (content['image_url']['url'] for content in message['content'] if content['type'] == 'image_url'), "")
            # Extract text content, find first, default if not match
            text = next(
                (content['text'] for content in message['content'] if content['type'] == 'text'), "")

            reformatted_content = f'{text} <img src="{image_url}" />'

            # Append to reformatted list
            reformatted_messages.append({"role": message['role'], "content": reformatted_content})

        return reformatted_messages

    @staticmethod
    def extract_content(line):
        output = {"content": "", "entities": []}
        choices = line.get("choices", [{}])
        choices = choices[0]
        if "message" in choices:
            content = choices.get("message", {}).get("content", "")
            entities = choices.get("message", {}).get("entities", [])
            output = {"content": content, "entities": entities}
        else:
            logger.warning("Message not found in response's json")

        return output

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
        messages = self.process_chat_messages(messages)

        if model_type == 'instruct' or model_type == 'chat':
            full_answer = self.call_chat_model(messages=messages, client=client)
        elif model_type == 'reward':
            full_answer = self.call_reward_model(messages=messages, client=client)
        elif model_type == 'coding':
            full_answer = self.call_coding_model(messages=messages, client=client)
        else:
            raise ValueError(f"Model type {model_type} for model {self.nim_model_name} is not supported")

        return full_answer

    def call_reward_model(self, messages, client):
        reward_dict = dict()
        if messages[-1].get('role') == 'assistant':
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages[1:],  # Remove default system prompt
            )
            rewards = completion.choices[0].logprobs.content
            for reward in rewards:
                reward_dict[reward.token] = reward.logprob
        return str(reward_dict)

    def call_chat_model(self, messages, client):
        if self.json_schema is not None:
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_token,
                stream=self.stream,
                extra_body={"nvext": {"guided_json": self.json_schema}}
            )
        else:
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_token,
                stream=self.stream
            )

        if self.stream is True:
            for chunk in completion:
                yield chunk.choices[0].delta.content or ""
        else:
            yield completion.choices[0].message.content or ""

    def call_coding_model(self, messages, client):
        code = client.completions.create(
            model=self.nim_model_name,
            prompt=messages[-1].get('content', ''),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            stream=self.stream
        )
        if self.stream is True:
            for chunk in code:
                yield chunk.choices[0].text or ""
        else:
            yield code.choices[0].text

    def call_multimodal(self, messages):
        url = f"https://ai.api.nvidia.com/v1/{self.nim_invoke_url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "messages": messages,
            "max_tokens": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }
        if self.nim_invoke_url != self.nim_model_name:
            payload["model"] = self.nim_model_name
        if self.json_schema is not None:
            payload["nvext"] = {"guided_json": self.json_schema}
        response = requests.post(url=url, headers=headers, json=payload, stream=self.stream)
        if not response.ok:
            raise ValueError(f'error:{response.status_code}, message: {response.text}')

        if self.stream is True:
            with response:  # To properly closed The response object after the block of code is executed
                logger.info("Streaming the response")
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        decoded_line = json.loads(line)
                        yield self.extract_content(decoded_line) or ""
        else:
            yield response.json().get('choices')[0].get('message').get('content')

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        add_metadata = self.configuration.get("add_metadata")
        for prompt_item in batch:
            # Get all messages including model annotations
            # For reward model
            model_name = self.configuration.get("model_to_reward", self.model_entity.name)

            messages = prompt_item.to_messages(model_name=model_name)
            if system_prompt != '':
                messages.insert(0, {"role": "system",
                                    "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            if self.nim_invoke_url.startswith('vlm/') or self.nim_invoke_url.startswith('gr/'):
                # VLM (Vision Language Model) - Multimodal models
                messages = self.process_multimodal_messages(messages)
                stream_response = self.call_multimodal(messages=messages)
            else:
                stream_response = self.call_model_open_ai(messages=messages)

            response = ""
            for chunk in stream_response:
                #  Multimodal responses
                if isinstance(chunk, dict):
                    entities = chunk.get("entities")
                    chunk = chunk.get("content")
                    if entities != []:
                        logger.info(f"Found {len(entities)} Bounding Boxes!")
                        elements = prompt_item.prompts[0].elements
                        image_item_id = (
                            next((element['value'].split('/')[-2] for element in elements if
                                  element['mimetype'] == 'image/*'), ""))
                        image_item = dl.items.get(item_id=image_item_id)
                        image_annotations = dl.AnnotationCollection()
                        for entity in entities:
                            label = entity.get("phrase")
                            bbox = entity.get("bboxes")[0]  # kosmos-2 expect one image for VQA task
                            image_annotations.add(
                                annotation_definition=dl.Box(left=bbox[0] * image_item.width,
                                                             top=bbox[1] * image_item.height,
                                                             right=bbox[2] * image_item.width,
                                                             bottom=bbox[3] * image_item.height,
                                                             label=label),

                                model_info={'name': self.model_entity.name,
                                            'model_id': self.model_entity.id,
                                            'confidence': 1.0})
                        image_item.annotations.upload(image_annotations)
                        logger.info("Uploaded bounding box annotations")

                # Other models responses
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})
        return []


if __name__ == '__main__':
    import dotenv

    dotenv.load_dotenv()

    dl.setenv('prod')
    project = dl.projects.get(project_name="InspectionAnalyticsDemo")
    model = dl.models.get(model_name="llama-3-2-90b-vision-instruct")
    dataset = project.datasets.get("TrialMLSImagery")
    item = dataset.items.get(item_id="67b9be75396c561cd8fefb1f")
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
