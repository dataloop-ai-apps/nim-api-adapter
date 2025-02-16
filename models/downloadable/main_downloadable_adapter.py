import dtlpy as dl
import subprocess
import logging
import time
from openai import OpenAI
import socket
import os

logger = logging.getLogger('NiM-Model')


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key: NGC_API_KEY")
        logger.info('Starting inference server')
        run_api_server_command = 'bash /opt/nim/start-server.sh'
        run_api_server = subprocess.Popen(run_api_server_command,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          shell=True)

        max_retries = 20
        while max_retries > 0 and self.is_port_available(host='0.0.0.0', port=8000) is True:
            logger.info(f'Waiting for inference server to start sleep {max_retries}')
            time.sleep(60 * 5)
            max_retries -= 1
            (out, err) = run_api_server.communicate()
            logger.info(f'Inference server output: {out}')
            logger.info(f'Inference server error: {err}')
        logger.info('Done Trying')
        if self.is_port_available(host='0.0.0.0', port=8000) is True:
            raise Exception('Unable to start inference server')

        self.client = OpenAI(base_url='http://0.0.0.0:8000/v1', api_key="not-used")

        self.nim_model_name = self.configuration.get('nim_model_name', None)
        if self.nim_model_name is None:
            raise Exception('Model name is missing in configuration')

    @staticmethod
    def is_port_available(host, port):
        """Checks if a port is available on a given host.

        Args:
            host: The hostname or IP address of the host.
            port: The port number to check.

        Returns:
            True if the port is available, False otherwise.
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            return False

    def call_model_open_ai(self, prompt):

        completion = self.client.completions.create(
            model=self.nim_model_name,
            prompt=prompt,
            max_tokens=1024,
            stream=False
        )
        full_answer = completion.choices[0].text
        return full_answer

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item=item)
        return prompt_item

    def predict(self, batch, **kwargs):
        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            full_answer = self.call_model_open_ai(prompt=messages[-1]['content'][0]['text'])
            prompt_item.add(message={"role": "assistant",
                                     "content": [{"mimetype": dl.PromptType.TEXT,
                                                  "value": full_answer}]},
                            model_info={'name': self.model_entity.name,
                                        'confidence': 1.0,
                                        'model_id': self.model_entity.id})
        return []
