import subprocess
import logging
import time
import socket

import os
import select
import threading
import json
import dtlpy as dl

logger = logging.getLogger("NiM-Model")

class BaseDownloadableModel(dl.BaseModelAdapter):
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

        logger.info("Model loaded successfully")

        self.load_model_config()

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


    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item=item)
        return prompt_item
    def predict(self, batch, **kwargs):
        # Implement prediction logic here
        pass
        
    def save(self, local_path, **kwargs):
        # Implement save logic here
        pass
        
    def embed(self, batch, **kwargs):
        # Implement embedding logic here
        pass
        
    def convert_from_dtlpy(self, data_path, **kwargs):
        # Implement conversion logic here
        pass

    def train(self, data_path, output_path, **kwargs):
        # Implement training logic here
        pass

    def load_model_config(self):
        pass