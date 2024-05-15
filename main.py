import requests, base64
import json
import dtlpy as dl
import logging

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        super().__init__(model_entity)
        with open('.env') as f:
            params = json.load(f)
        self.invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
        self.headers = {
            "Authorization": f"Bearer {params['api_key']}",
            "Accept": "application/json"
        }

    def prepare_item_func(self, item: dl.Item):
        if 'json' not in item.mimetype:
            logger.warning(f"Item is not a JSON file.")
            return None
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def predict(self, batch, **kwargs):
        annotations = []
        for prompt_item in batch:
            ann_collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                text = None
                encoded_image = None
                for partial_prompt in prompt_content:
                    if 'image' in partial_prompt.get('mimetype', ''):
                        image_url = partial_prompt.get('value', '')
                        item_id = image_url.split("/stream")[0].split("/items/")[-1]
                        image_buffer = dl.items.get(item_id=item_id).download(save_locally=False).getvalue()
                        encoded_image = base64.b64encode(image_buffer).decode()

                        # Encoding testing
                        # decoded_bytes = base64.b64decode(encoded_image)
                        # Open a file for writing in binary mode and write the decoded bytes
                        # with open("image.jpg", "wb") as image_file:
                        #     image_file.write(decoded_bytes)

                    elif 'text' in partial_prompt.get('mimetype', ''):
                        text = partial_prompt.get('value')
                    else:
                        logger.warning(f"Prompt is missing either an image or a text prompt.")
                if text is None or encoded_image is None:
                    logger.warning(f"{prompt_name} is missing either an image or a text prompt.")
                    continue
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f'{text}. <img src="data:image/png;base64,{encoded_image}" />'
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.20,
                    "top_p": 0.70,
                    "seed": 0,
                    "stream": False
                }
                response = requests.post(self.invoke_url, headers=self.headers, json=payload)
                content = response.json().get('choices')[0].get('message').get('content')
                ann_collection.add(
                    annotation_definition=dl.FreeText(text=content),
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
