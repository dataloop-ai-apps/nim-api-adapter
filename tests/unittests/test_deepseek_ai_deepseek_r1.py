import json

import dotenv
import unittest
from models.api.main_api_adapter import ModelAdapter
import dtlpy as dl

dotenv.load_dotenv('.env')


# Define a test case class
class TestModelAdapter(unittest.TestCase):

    def test_inference(self):
        with open("models/api/deepseek_ai/deepseek_r1/dataloop.json") as f:
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
        stream_response = adapter.call_model_open_ai(messages)
        response = ""
        for chunk in stream_response:
            response += chunk
        print(f"model `{dummy_model.name}`, {adapter.nim_model_name}. output: {response}")


if __name__ == '__main__':
    unittest.main()
