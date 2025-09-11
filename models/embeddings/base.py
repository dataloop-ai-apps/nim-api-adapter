import base64
import io
import logging
import os
import select
import socket
import subprocess
import threading
import time

import dtlpy as dl
import numpy as np
from openai import OpenAI
from PIL import Image

logger = logging.getLogger("NIM Adapter")

class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        ngc_api_key = os.environ.get("NGC_API_KEY", None)
        if ngc_api_key is None:
            raise ValueError("Missing API key")
        self.api_key = ngc_api_key
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

        self.is_downloadable = self.configuration.get("is_downloadable", False)
        if self.is_downloadable:
            self.base_url = "http://0.0.0.0:8000/v1"
            ModelAdapter.start_and_wait_for_server()
        else:
            self.base_url = "https://integrate.api.nvidia.com/v1"

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def call_model_open_ai(self, text):        
        response = self.client.embeddings.create(
            input=[text],
            model=self.nim_model_name,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        embedding = response.data[0].embedding
        return embedding

    def call_model_open_ai_batch(self, inputs_list):
        response = self.client.embeddings.create(
            input=inputs_list,
            model=self.nim_model_name,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return [d.embedding for d in response.data]

    def embed(self, batch, **kwargs):
        input_list = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            elif isinstance(item, np.ndarray):
                text = ModelAdapter.np_to_base64(item)
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][0]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][0]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text , image or prompt items are supported {e}')
            input_list.append(text)
        embedding = self.call_model_open_ai_batch(input_list)
        for item, embedding in zip(batch, embedding):
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
        return embedding

    @staticmethod
    def np_to_base64(img_array: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(img_array.astype("uint8")).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        free = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        total = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        used = [int(x.split()[0]) for i, x in enumerate(info)]
        return free, total, used

    @staticmethod
    def keep():
        while True:
            free, total, used = ModelAdapter.get_gpu_memory()
            logger.info(f"gpu memory - total: {total}, used: {used}, free: {free}")
            time.sleep(5)

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
        

    @staticmethod
    def start_and_wait_for_server():
        threading.Thread(target=ModelAdapter.keep, daemon=True).start()

        logger.info("Starting inference server")
        run_api_server_command = "/bin/bash -c /opt/nim/start_server.sh"
        run_api_server = subprocess.Popen(
            run_api_server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )


        max_retries = 0
        while (
            max_retries < 20
            and ModelAdapter.is_port_available(host="0.0.0.0", port=8000) is True
        ):
            logger.info(
                f"Waiting for inference server to start sleep iteration {max_retries} sleeping for 5 minutes"
            )
            time.sleep(60 * 5)
            max_retries += 1
            logger.info(f"Still waiting current logs: ")
            readable, _, _ = select.select(
                [run_api_server.stdout, run_api_server.stderr], [], [], 0.1
            )
            for f in readable:
                line = f.readline()
                if line:
                    print(f"Output: {line.strip()}")
        logger.info("Done Trying")
        if ModelAdapter.is_port_available(host="0.0.0.0", port=8000) is True:
            raise Exception("Unable to start inference server")
        
        return True


