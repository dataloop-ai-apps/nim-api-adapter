import json
import base64
import dotenv
import unittest
from models.api.vision_models.vision_model_adapter import ModelAdapter
import dtlpy as dl

dotenv.load_dotenv(".env")


# Define a test case class
class TestModelAdapter(unittest.TestCase):

    def test_inference(self):
        with open("models/api/vision_models/baidu_paddleocr/dataloop.json") as f:
            manifest = json.load(f)
        with open("tests/assets/unittests/sample_image.png", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        model_json = manifest["components"]["models"][0]
        dummy_model = dl.Model.from_json(_json=model_json, client_api=dl.client_api, project=None, package=dl.Package())
        adapter = ModelAdapter(model_entity=dummy_model)
        adapter.load("./")
        output = adapter.call_model(image_b64)
        print(f"model `{dummy_model.name}`, {adapter.nim_invoke_url}. output: {output}")


if __name__ == "__main__":
    unittest.main()
