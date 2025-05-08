import dtlpy as dl
import numpy as np
import requests
import logging
import base64
import cv2
import os

logger = logging.getLogger("Vision NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key")

        self.adapter_defaults.upload_annotations = False

        self.api_key = os.environ.get("NGC_API_KEY", None)

        self.nim_invoke_url = self.configuration.get("nim_invoke_url")
        if self.nim_invoke_url is None:
            raise ValueError("nim_invoke_url is not set! Insert the nim url in the model configuration.")

    def prepare_item_func(self, item: dl.Item):
        buffer = item.download(save_locally=False)
        buffer.seek(0)  # Reset the buffer position to the beginning

        # Read the buffer content as a NumPy array and decode it to an OpenCV image
        file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Encode to base64
        buffer.seek(0)  # Reset buffer position again for encoding
        image_b64 = base64.b64encode(buffer.read()).decode()

        return item, img, image_b64  # Return both the OpenCV image and base64 string
    
    def call_model(self, image_b64, payload=None):
        url = f"https://ai.api.nvidia.com/v1/{self.nim_invoke_url}"
        assert len(image_b64) < 400_000, "To upload larger images, use the assets API (see docs)"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
            }

        if payload is None: # use default payload
            payload = {
                "input": [
                    {
                    "type": "image_url",
                    "url": f"data:image/png;base64,{image_b64}"
                    }
                ]
                }
        
        response = requests.post(url=url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to call model: {response.status_code} {response.text}")
        
        return response.json()
    
    def extract_annotations_yolox(self, img, image_b64, collection):
        response = self.call_model(image_b64)
        for box in response.get('data', []): 
            for label,values in box['bounding_boxes'].items():
                for value in values:
                    collection.add(annotation_definition=dl.Box(left=max(value['x_min']*img.shape[1], 0),
                                                                    top=max(value['y_min']*img.shape[0], 0),
                                                                    right=min(value['x_max']*img.shape[1], img.shape[1]),
                                                                    bottom=min(value['y_max']*img.shape[0], img.shape[0]),
                                                                    label=label
                                                                    ),
                                                model_info={'name': self.model_entity.name,
                                                'model_id': self.model_entity.id,
                                                'confidence': value['confidence']})
        return collection
    
    def extract_annotations_paddleocr(self, img, image_b64, collection):    
        response = self.call_model(image_b64)
        annotations = response.get('data', [])[0].get('text_detections', [])
        for annotation in annotations:
            text_annotation = annotation.get('text_prediction', {})
            label = text_annotation.get('text', '')
            confidence = text_annotation.get('confidence', 0)
            points = annotation.get('bounding_box',{}).get('points', [])
            x_min = min(point['x'] for point in points)
            y_min = min(point['y'] for point in points)
            x_max = max(point['x'] for point in points)
            y_max = max(point['y'] for point in points)
            collection.add(annotation_definition=dl.Box(left=max(x_min*img.shape[1], 0),
                                                            top=max(y_min*img.shape[0], 0),
                                                            right=min(x_max*img.shape[1], img.shape[1]),
                                                            bottom=min(y_max*img.shape[0], img.shape[0]),
                                                            description=label,
                                                            label="text"
                                                            ),
                                                model_info={'name': self.model_entity.name,
                                               'model_id': self.model_entity.id,
                                               'confidence': round(confidence, 3)})
        return collection
    
    def extract_annotations_cached(self, item, image_b64, collection):
        payload = {
                    "messages": [
                        {
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                            }
                        }]
                        }
                    ]
                    }
        
        response = self.call_model(image_b64, payload)
        detections = response.get('data', [])[0].get('content', {})
        # upload to metadata
        if 'user' not in item.metadata:
            item.metadata['user'] = {}
        item.metadata['user']['cached_response'] = detections
        item.update(True)
        
        return collection
    
    def predict(self, batch, **kwargs):
        batch_annotations = list()
        for item, img, image_b64 in batch:
            collection = dl.AnnotationCollection()
            if 'nv-yolox-page-elements-v1' in self.model_entity.name:
                collection = self.extract_annotations_yolox(img, image_b64, collection)
            elif 'baidu-paddleocr' in self.model_entity.name:
                collection = self.extract_annotations_paddleocr(img, image_b64, collection)
            elif 'university-at-buffalo-cached' in self.model_entity.name:
                collection = self.extract_annotations_cached(item, image_b64, collection)
            else:
                raise ValueError(f"Model {self.model_entity.name} not supported")
            
            batch_annotations.append(collection)

        return batch_annotations
            
