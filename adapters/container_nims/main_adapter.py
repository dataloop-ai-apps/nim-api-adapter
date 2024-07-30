import dtlpy as dl
import subprocess
import logging
import time
from openai import OpenAI

logger = logging.getLogger('NiM-Model')


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        run_api_server_command = 'bash /opt/nim/start-server.sh'
        run_api_server = subprocess.Popen(run_api_server_command,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          shell=True)
        while run_api_server.poll() is None:
            time.sleep(120)

        (out, err) = run_api_server.communicate()
        if run_api_server.returncode != 0:
            raise Exception(f'Failed to start API server: {err}')
        self.client = OpenAI(base_url='http://0.0.0.0:8000/v1', api_key="not-used")

        self.nim_model_name = self.configuration.get('model_name', None)
        if self.nim_model_name is None:
            raise Exception('Model name is missing in configuration')

    def call_model_chat_open_ai(self, messages):

        completion = self.client.chat.completions.create(
            model=self.nim_model_name,
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False
        )
        full_answer = completion.choices[0].message.content
        return full_answer

    def call_model_open_ai(self, prompt):

        completion = self.client.completions.create(
            model=self.nim_model_name,
            prompt=prompt,
            max_tokens=1024,
            stream=False
        )
        full_answer = completion.choices[0].text
        return full_answer

    @staticmethod
    def process_messages(messages):
        for message in messages:
            message_string = ""
            for element in message.get('content', []):
                message_string += element.get('text', "")
            message['content'] = message_string
        return messages

    def predict(self, batch, **kwargs):
        for prompt_item in batch:
            messages = prompt_item.messages(model_name=self.model_entity.name)
            try:
                full_answer = self.call_model_chat_open_ai(messages=messages)
                print('heeeeeeeeeereee')
            except:
                print('faaaaaaaaaaaaaail')
                full_answer = self.call_model_open_ai(prompt=self.process_messages(messages[-1]['content']))
            annotation = dl.FreeText(text=full_answer)
            prompt_item.add_responses(annotation=annotation, model=self.model_entity)


if __name__ == '__main__':
    dl.setenv('rc')
    model = dl.models.get(model_id='')
    item = dl.items.get(item_id='')
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
