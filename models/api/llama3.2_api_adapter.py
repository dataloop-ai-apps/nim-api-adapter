from Tools.scripts.parse_html5_entities import entities_url
from openai import OpenAI
import dtlpy as dl
import requests
import logging
import json
import os
from main_api_adapter import ModelAdapter

logger = logging.getLogger("NIM Adapter")


class LlamaAdapter(ModelAdapter):
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
        if self.guided_json is not None:
            payload['messages'][0]['content'] += f" {self.guided_json}"
            payload["nvext"] = {"guided_json": self.guided_json}
        logger.info(f"Payload sent to model: {payload}")
        response = requests.post(url=url, headers=headers, json=payload)
        if not response.ok:
            raise ValueError(f'error:{response.status_code}, message: {response.text}')

        if self.stream is True:
            with response:  # To properly closed The response object after the block of code is executed
                logger.info("Streaming the response")
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8').replace("data: ", "")
                        if "[DONE]" not in line:
                            decoded_line = json.loads(line)
                        yield self.extract_content(decoded_line) or ""
        else:
            yield response.json().get('choices')[0].get('message').get('content')

    @staticmethod
    def extract_content(line):
        output = {"content": "", "entities": []}
        choices = line.get("choices", [{}])
        choices = choices[0]
        if "delta" in choices:
            content = choices.get("delta", {}).get("content", "")
            entities = []
            output = {"content": content, "entities": entities}
        else:
            logger.warning("Message not found in response's json")

        return output



if __name__ == '__main__':
    import dotenv

    dotenv.load_dotenv()

    dl.setenv('prod')
    project = dl.projects.get(project_name="InspectionAnalyticsDemo")
    model = project.models.get(model_name="llama-3-2-90b-vision-instruct")
    dataset = project.datasets.get("TrialMLSImagery")
    # model.configuration['stream'] = True
    item = dataset.items.get(item_id="67bc772b13b6fb55b48db731")

    anns = item.annotations.list().delete()

    adapter =LlamaAdapter(model)
    adapter.predict_items(items=[item])
