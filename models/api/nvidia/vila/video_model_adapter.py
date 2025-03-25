import os
import sys

# Add the project root directory to the Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
)

import json
import logging
import dtlpy as dl
from models.api.main_api_adapter import ModelAdapter

logger = logging.getLogger("NIM Adapter")


class VideoModelAdapter(ModelAdapter):
    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get("system_prompt", "")
        add_metadata = self.configuration.get("add_metadata")
        for prompt_item in batch:
            # Get all messages including model annotations
            model_name = self.configuration.get(
                "model_to_reward", self.model_entity.name
            )

            messages = prompt_item.to_messages(model_name=model_name)
            if system_prompt != "":
                messages.insert(0, {"role": "system", "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get("nearestItems", [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(
                    nearest_items=nearest_items, add_metadata=add_metadata
                )
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            if self.nim_invoke_url.startswith("vlm/") or self.nim_invoke_url.startswith(
                "gr/"
            ):
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
                        image_item_id = next(
                            (
                                element["value"].split("/")[-2]
                                for element in elements
                                if element["mimetype"] == "image/*"
                            ),
                            "",
                        )
                        image_item = dl.items.get(item_id=image_item_id)
                        image_annotations = dl.AnnotationCollection()
                        for entity in entities:
                            label = entity.get("phrase")
                            bbox = entity.get("bboxes")[
                                0
                            ]  # kosmos-2 expect one image for VQA task
                            image_annotations.add(
                                annotation_definition=dl.Box(
                                    left=bbox[0] * image_item.width,
                                    top=bbox[1] * image_item.height,
                                    right=bbox[2] * image_item.width,
                                    bottom=bbox[3] * image_item.height,
                                    label=label,
                                ),
                                model_info={
                                    "name": self.model_entity.name,
                                    "model_id": self.model_entity.id,
                                    "confidence": 1.0,
                                },
                            )
                        image_item.annotations.upload(image_annotations)
                        logger.info("Uploaded bounding box annotations")

                # Other models responses
                response += chunk
                prompt_item.add(
                    message={
                        "role": "assistant",
                        "content": [
                            {"mimetype": dl.PromptType.TEXT, "value": response}
                        ],
                    },
                    model_info={
                        "name": self.model_entity.name,
                        "confidence": 1.0,
                        "model_id": self.model_entity.id,
                    },
                )
        return []


if __name__ == "__main__":
    dl.setenv("rc")
    # with open("models/api/nvidia/vila/dataloop.json") as f:
    with open(r"C:\Users\Yaya Tang\PycharmProjects\nim-api-adapter\models\api\nvidia\vila\dataloop.json") as f:
        manifest = json.load(f)
    model = dl.Model.from_json(
        _json=manifest["components"]["models"][0],
        client_api=dl.client_api,
        project=None,
        package=dl.Package(),
    )

    project = dl.projects.get(project_name="Model mgmt demo")
    dataset = project.datasets.get(dataset_name="llama_testing")
    item = dataset.items.get(item_id="67e2aa938d574df4e1c299c9")
    item.annotations.list().delete()

    adapter = ModelAdapter(model)
    items, annotations = adapter.predict_items(items=[item])

    print(annotations)
