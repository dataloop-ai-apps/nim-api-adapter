import json

import dotenv
import unittest
from models.api_nim.main_adapter import ModelAdapter
import dtlpy as dl

dotenv.load_dotenv('.env')


# Define a test case class
class TestModelAdapter(unittest.TestCase):

    def test_inference(self):
        with open("api_nim/nvidia_neva_22b/dataloop.json") as f:
            manifest = json.load(f)
        model_json = manifest['components']['models'][0]
        dummy_model = dl.Model.from_json(_json=model_json,
                                         client_api=dl.client_api,
                                         project=None,
                                         package=dl.Package())
        adapter = ModelAdapter(model_entity=dummy_model)
        adapter.load('./')
        messages = [{"role": "user",
                     "content": "What is the most important thing a hitchhiker can carry?"}]
        output = adapter.call_model_requests(messages)
        print(f"model `{dummy_model.name}`, {adapter.nim_model_name}. output: {output}")


if __name__ == '__main__':
    unittest.main()
