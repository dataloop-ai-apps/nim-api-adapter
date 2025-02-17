import os
import json
import dotenv
import unittest
import dtlpy as dl
from pydantic_core.core_schema import json_schema

from models.api.main_api_adapter import ModelAdapter

dotenv.load_dotenv('.env')

json_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The short answer or response."
        },
        "explanation": {
            "type": "string",
            "description": "A detailed explanation for the answer."
        }
    },
    "required": ["answer", "explanation"],
    "additionalProperties": False
}


# Define a test case class
class TestModelAdapter(unittest.TestCase):

    def test_inference(self):
        file_path = os.path.abspath("models/api/meta/llama_3_2_90b_vision_instruct/dataloop.json")
        with open(file_path) as f:
            manifest = json.load(f)
        model_json = manifest['components']['models'][0]
        dummy_model = dl.Model.from_json(_json=model_json,
                                         client_api=dl.client_api,
                                         project=None,
                                         package=dl.Package())

        adapter = ModelAdapter(model_entity=dummy_model)
        adapter.stream = False
        adapter.json_schema = json.loads(json_schema)

        adapter.load('./')
        messages = [{"role": "user",
                     "content": "What is the most important thing a hitchhiker can carry?"}]
        output = adapter.call_model_open_ai(messages)
        print(f"model `{dummy_model.name}`,\n\n"
              f"{adapter.nim_model_name}.\n\n"
              f" output: {output}")


if __name__ == '__main__':
    unittest.main()
