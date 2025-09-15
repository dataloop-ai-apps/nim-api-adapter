import dtlpy as dl
import requests
import subprocess
import logging
import time
import socket
import os
import select
import threading
import base64
import numpy as np
import cv2
logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError("Missing API key")

        self.api_key = os.environ.get("NGC_API_KEY", None)
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")
        self.base_url = "http://0.0.0.0:8000/v1"
        self.start_and_wait_for_server()

    def embed(self, batch, **kwargs):
        """
        Generate embeddings for a batch of items (text or images).
        Sends HTTP POST request to local server for embeddings.
        """
        # Prepare input list for the embeddings API
        input_list = []

        for item in batch:
            if isinstance(item, str):
                # Direct text input
                input_list.append(item)
            elif isinstance(item, np.ndarray):
                # Image numpy array - convert to base64
                try:
                    # Convert numpy array to image format
                    if len(item.shape) == 3 and item.shape[2] == 3:
                        # RGB image
                        image = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
                    elif len(item.shape) == 2:
                        # Grayscale image
                        image = item
                    else:
                        logger.warning(f"Unsupported image shape: {item.shape}")
                        continue

                    # Encode image to base64
                    success, buffer = cv2.imencode('.png', image)
                    if success:
                        base64_data = base64.b64encode(buffer).decode('utf-8')
                        data_url = f"data:image/png;base64,{base64_data}"
                        input_list.append(data_url)
                    else:
                        logger.warning("Failed to encode image to PNG")
                        continue
                except Exception as e:
                    logger.warning(f"Could not convert numpy array to base64: {e}")
                    continue
            else:
                logger.warning(f"Unsupported item type: {type(item)}")
                continue

        if not input_list:
            logger.warning("No valid items found in batch for embedding")
            return []
        logger.info(f"Input list: {input_list}")
        # Make HTTP POST request to embeddings API
        try:
            url = f"{self.base_url}/embeddings"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

            payload = {"input": input_list, "model": self.nim_model_name, "encoding_format": "float"}

            logger.info(f"Sending embeddings request to {url} with {len(input_list)} items")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if not response.ok:
                raise ValueError(f"Embeddings API error: {response.status_code}, message: {response.text}")

            response_data = response.json()
            embeddings = []

            # Extract embeddings from response
            if 'data' in response_data:
                for embedding_data in response_data['data']:
                    if 'embedding' in embedding_data:
                        embeddings.append(embedding_data['embedding'])
                    else:
                        logger.warning("No embedding found in response data")
                        embeddings.append([])
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return []

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error calling embeddings API: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}")

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

    def start_and_wait_for_server(self):
        """
        Start the inference server and wait for it to start.
        """
        threading.Thread(target=ModelAdapter.keep, daemon=True).start()

        logger.info("Starting inference server")
        run_api_server_command = "bash /opt/nim/start-server.sh"
        run_api_server = subprocess.Popen(
            run_api_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
        )

        max_retries = 0
        while max_retries < 20 and ModelAdapter.is_port_available(host="0.0.0.0", port=8000) is True:
            logger.info(f"Waiting for inference server to start sleep iteration {max_retries} sleeping for 5 minutes")
            time.sleep(60*5)
            max_retries += 1
            logger.info("Still waiting current logs: ")
            readable, _, _ = select.select([run_api_server.stdout, run_api_server.stderr], [], [], 0.1)
            for f in readable:
                line = f.readline()
                if line:
                    print(f"Output: {line.strip()}")
        logger.info("Done Trying")
        if ModelAdapter.is_port_available(host="0.0.0.0", port=8000) is True:
            raise Exception("Unable to start inference server")

        return True