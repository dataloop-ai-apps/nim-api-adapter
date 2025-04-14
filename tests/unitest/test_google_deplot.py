import json
import base64
import dotenv
import unittest
from models.api.custom_model_adapter import CustomModelAdapter
import dtlpy as dl

dotenv.load_dotenv(".env")


# Define a test case class
class TestModelAdapter(unittest.TestCase):

    def test_inference(self):
        with open("models/api/google/deplot/dataloop.json") as f:
            manifest = json.load(f)
        with open("test/assets/unittests/sample_image.png", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        model_json = manifest["components"]["models"][0]
        dummy_model = dl.Model.from_json(_json=model_json, client_api=dl.client_api, project=None, package=dl.Package())
        adapter = CustomModelAdapter(model_entity=dummy_model)
        adapter.load("./")
        messages = [
            {
                "role": "user",
                "content": (
                    f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
                ),
            }
        ]
        output = adapter.call_multimodal(messages)
        print(f"model `{dummy_model.name}`, {adapter.nim_model_name}. output: {output}")


if __name__ == "__main__":
    unittest.main()
