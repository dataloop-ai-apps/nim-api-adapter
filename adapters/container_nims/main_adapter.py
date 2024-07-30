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
