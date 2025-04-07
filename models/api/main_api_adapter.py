from openai import OpenAI
import dtlpy as dl
import requests
import logging
import base64
import json
import os
import re

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError("Missing API key")

        self.adapter_defaults.upload_annotations = False

        self.api_key = os.environ.get("NGC_API_KEY", None)
        self.max_token = self.configuration.get('max_token', 1024)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.7)
        self.seed = self.configuration.get('seed', None)
        self.stream = self.configuration.get('stream', True)
        self.num_frames_per_inference = self.configuration.get('num_frames_per_inference', None)

        self.guided_json = self.configuration.get("guided_json", None)
        if self.guided_json is not None:
            try:
                item = dl.items.get(item_id=self.guided_json)
                binaries = item.download(save_locally=False)
                self.guided_json = json.loads(binaries.getvalue().decode("utf-8"))
                logger.info(f"Guided json: {self.guided_json}")
            except Exception as e:  # noqa: F841
                try:
                    self.guided_json = json.loads(self.guided_json)
                except Exception as e:
                    logger.error(f"Error loading guided json: {e}")

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

            # Check if there is a video url in the text
            text, video_url = ModelAdapter.check_video_url(text)

            if video_url is not None:
                reformatted_content = (f'{text} <video src="{video_url}" />')
            else:
                reformatted_content = (f'{text} <img src="{image_url}" />')

            # Append to reformatted list
            reformatted_messages.append(
                {"role": message["role"], "content": reformatted_content}
            )

        return reformatted_messages

    @staticmethod
    def check_video_url(text: str):
        """
        Extracts first URL from a given text string.

        :param text: The input text.
        :return: The cleaned text and the video url.
        """
        clean_text = text
        video_b64 = None

        url_pattern = r"https?://[^\s)]+"
        links = re.findall(url_pattern, text)
        for link in links:
            if "gate.dataloop.ai/api/v1/items/" in link:
                try:
                    clean_text = clean_text.replace(link, "")
                    # Extract item ID from URL after "items/"
                    item_id = link.split("items/")[1].split("/")[0]
                    item = dl.items.get(item_id=item_id)
                    if item.mimetype == "video/mp4":
                        binaries = item.download(save_locally=False)
                        buffer= binaries.getvalue()
                        video_b64 = base64.b64encode(buffer).decode('utf-8')
                        video_b64 = f"data:video/mp4;base64,{video_b64}"
                    else:
                        logger.error(f"Video item type must be mp4, got {item.mimetype} for link: {link}")
                except Exception as e:
                    logger.error(f"Error downloading video: {e}. Ignoring link: {link}")
        return clean_text, video_b64

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
        if self.guided_json is not None:
            completion = client.chat.completions.create(
                model=self.nim_model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_token,
                stream=self.stream,
                extra_body={"nvext": {"guided_json": self.guided_json}}
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
            "Content-Type": "application/json",
            "accept": "application/json",
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
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.num_frames_per_inference is not None:
            payload["num_frames_per_inference"] = self.num_frames_per_inference
        if self.guided_json is not None:
            payload["nvext"] = {"guided_json": self.guided_json}
        logger.info(f"Payload sent to model: {payload}")
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


if __name__ == "__main__":
    dl.setenv("rc")
    with open("models/api/nvidia/vila/dataloop.json") as f:
        manifest = json.load(f)
    model = dl.Model.from_json(_json=manifest["components"]["models"][0], client_api=dl.client_api, project=None, package=dl.Package()) 
    model.configuration["seed"] = None
    
    project = dl.projects.get(project_name="Model mgmt demo")
    dataset = project.datasets.get(dataset_name="llama_testing")
    item = dataset.items.get(item_id="67e409c811a30628e9b8b85d") # dasgh....json
    item.annotations.list().delete()

    adapter = ModelAdapter(model)
    items, annotations = adapter.predict_items(items=[item])

    print(annotations)