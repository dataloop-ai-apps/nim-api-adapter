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
        model_path = os.path.abspath("models/api/meta/llama_3_2_90b_vision_instruct/dataloop.json")
        with open(model_path) as f:
            manifest = json.load(f)
        model_json = manifest['components']['models'][0]
        dummy_model = dl.Model.from_json(_json=model_json,
                                         client_api=dl.client_api,
                                         project=None,
                                         package=dl.Package())
        dummy_model.configuration['stream'] = False
        dummy_model.configuration['json_schema'] = json_schema
        adapter = ModelAdapter(model_entity=dummy_model)

        adapter.load('./')
        # prompt_file = os.path.abspath("test/assets/unittests/prompt_item.json")
        # prompt_item = dl.PromptItem.from_local_file(filepath=prompt_file)
        # output = adapter.predict([prompt_item])
        messages = [{"role": "user",
                     "content": "What is the most important thing a hitchhiker can carry?"}]
        output = adapter.call_multimodal(messages)

        print(f"model `{dummy_model.name}`,\n\n"
              f"{adapter.nim_model_name}.\n\n"
              f" output: {[print(out) for out in output]}")
        print()



if __name__ == '__main__':
    unittest.main()
