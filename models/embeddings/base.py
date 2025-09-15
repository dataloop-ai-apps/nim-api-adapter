from openai import OpenAI
import dtlpy as dl
import subprocess
import time
import socket
import select
import threading
import logging
import os

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing API key")
        self.api_key = os.environ.get("NGC_API_KEY", None)
        print(f'-HHH- api key: {self.api_key}')
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")

        self.is_downloadable = self.configuration.get("is_downloadable", False)
        if self.is_downloadable:
            self.base_url = "http://0.0.0.0:8000/v1"
            self.start_and_wait_for_server()
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

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
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
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            embedding = self.call_model_open_ai(text)
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings

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
        

    def start_and_wait_for_server(self):
        threading.Thread(target=self.keep, daemon=True).start()

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
        
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    
    print("start")
    use_rc_env = False
    if use_rc_env:
        dl.setenv('rc')
    else:
        dl.setenv('prod')
    # dl.logout()
    if dl.token_expired():
        dl.login()
    print("login done")
    proejct  = dl.projects.get(project_name="ShadiDemo")
    model_entity = proejct.models.get(model_name="rf-detr-abd0d")
    model_entity.configuration['nim_model_name'] = "nvidia/nvclip"
    print("-HHH- 1")
    model = ModelAdapter(model_entity)
    print("-HHH- 2")
    dataset = proejct.datasets.get(dataset_id="680c97da8580d18187236c95")
    print("-HHH- 3")
    # dataset.open_in_web()
    model.embed_dataset(dataset=dataset)
    print("-HHH- 4")