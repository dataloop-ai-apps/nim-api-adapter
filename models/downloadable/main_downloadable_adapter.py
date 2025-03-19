import dtlpy as dl
import subprocess
import logging
import time
from openai import OpenAI
import socket
import requests
import os
import select
import threading
import json

logger = logging.getLogger("NiM-Model")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key: NGC_API_KEY")

        threading.Thread(target=self.keep, daemon=True).start()

        logger.info("Starting inference server")
        run_api_server_command = "bash /opt/nim/start-server.sh"
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
            and self.is_port_available(host="0.0.0.0", port=8000) is True
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
        if self.is_port_available(host="0.0.0.0", port=8000) is True:
            raise Exception("Unable to start inference server")

        self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")

        self.nim_model_name = self.configuration.get("nim_model_name", None)
        if self.nim_model_name is None:
            raise Exception("Model name is missing in configuration")
        self.guided_json = self.configuration.get("guided_json", None)
        if self.guided_json is not None:
            try:
                item = dl.items.get(item_id=self.guided_json)
                binaries = item.download(save_locally=False)
                self.guided_json = json.loads(binaries.getvalue().decode("utf-8"))
                logger.info(f"Guided json: {self.guided_json}")
            except Exception as e:
                logger.error(f"Error loading guided json: {e}")

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

    def keep(self):
        while True:
            free, total, used = self.get_gpu_memory()
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

    def call_model_open_ai(self, prompt):
        completion = self.client.completions.create(
            model=self.nim_model_name, prompt=prompt, max_tokens=1024, stream=False
        )
        full_answer = completion.choices[0].text
        return full_answer

    def call_model_requests(self, messages):
        seed = self.configuration.get("seed", 20)
        temperature = self.configuration.get("temperature", 0)
        max_tokens = self.configuration.get("max_tokens", 256)
        url = "http://0.0.0.0:8000/v1/chat/completions"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {
            "model": "meta/llama-3.2-11b-vision-instruct",
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if self.guided_json is not None:
            data["nvext"] = {"guided_json": self.guided_json}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        full_answer = response_json["choices"][0]["message"]["content"]
        return full_answer

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item=item)
        return prompt_item

    def predict(self, batch, **kwargs):
        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            if self.configuration.get("request_type", "openai") == "openai":
                full_answer = self.call_model_open_ai(
                    prompt=messages[-1]["content"][0]["text"]
                )
            else:
                full_answer = self.call_model_requests(messages)
            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": full_answer}],
                },
                model_info={
                    "name": self.model_entity.name,
                    "confidence": 1.0,
                    "model_id": self.model_entity.id,
                },
            )
        return []
