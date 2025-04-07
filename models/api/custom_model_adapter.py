import json
import logging
import requests
import dtlpy as dl
from models.api.main_api_adapter import ModelAdapter

logger = logging.getLogger("NIM Adapter")


class CustomModelAdapter(ModelAdapter):
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
            "stream": self.stream,
            "seed": self.seed,
        }
        if self.nim_invoke_url != self.nim_model_name:
            payload["model"] = self.nim_model_name
        if self.num_frames_per_inference is not None:
            payload["num_frames_per_inference"] = self.num_frames_per_inference
        if self.guided_json is not None:
            payload["messages"][0]["content"] += f" {self.guided_json}"
            payload["nvext"] = {"guided_json": self.guided_json}
        logger.info(f"Payload sent to model: {payload}")
        response = requests.post(url=url, headers=headers, json=payload)
        if not response.ok:
            raise ValueError(f"error:{response.status_code}, message: {response.text}")

        if self.stream is True:
            with response:  # To properly closed The response object after the block of code is executed
                logger.info("Streaming the response")
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        lookup_key = "delta" if line[0:4] == "data" else "messages"
                        line = line.replace("data: ", "")  # specific to llama3.2 vision instruct output
                        if "[DONE]" not in line:
                            decoded_line = json.loads(line)
                        yield self.extract_content(decoded_line, lookup_key) or ""
        else:
            yield response.json().get("choices")[0].get("message").get("content")

    @staticmethod
    def extract_content(line, response_key="messages"):
        output = {"content": "", "entities": []}
        choices = line.get("choices", [{}])
        choices = choices[0]
        if response_key in choices:
            content = choices.get(response_key, {}).get("content", "")
            entities = choices.get(response_key, {}).get("entities", [])
            output = {"content": content, "entities": entities}
        else:
            logger.warning("Message not found in response's json")

        return output


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    dl.setenv("prod")
    model = dl.models.get(model_id="")
    item = dl.items.get(item_id="")

    adapter = CustomModelAdapter(model)
    adapter.predict_items(items=[item])
