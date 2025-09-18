import os
import time
import logging
import subprocess
from typing import Optional
import io
import base64
from PIL import Image
import requests
import dtlpy as dl


logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    """Embeddings adapter that boots the NV-CLIP NIM server inside the container.

    In some execution environments the Docker ENTRYPOINT is not invoked. This
    adapter explicitly starts the NIM server (same command as the image entrypoint)
    and waits until the OpenAI-compatible endpoint is ready on port 8000.
    """

    _server_proc: Optional[subprocess.Popen] = None

    def load(self, local_path, **kwargs):
        api_key = os.environ.get("NGC_API_KEY", "")
        if not api_key:
            raise ValueError("Missing NGC_API_KEY")

        # Start the NIM server (equivalent to the container entrypoint)
        cmd = "bash /opt/nim/start_server.sh"
        # Enforce system Python 3.10 for any python invocations inside the script
        runtime_env = {
            **os.environ,
            "NGC_API_KEY": api_key,
            "PATH": f"/usr/bin:{os.environ.get('PATH', '')}",
            "PYTHON": "/usr/bin/python3",
            "PYTHONEXECUTABLE": "/usr/bin/python3",
        }

        # Sanity check the interpreter
        try:
            subprocess.run(["/usr/bin/python3", "-V"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            raise RuntimeError(f"/usr/bin/python3 not available or not working: {e}")

        logger.info("Starting NV-CLIP NIM server: %s", cmd)
        self._server_proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=runtime_env,
        )

        # Wait for readiness on the OpenAI-compatible endpoint
        base_url = "http://127.0.0.1:8000"
        ready = False
        for _ in range(300):  # up to ~300 seconds
            # If the process died early, surface logs
            if self._server_proc.poll() is not None:
                output = ""
                try:
                    if self._server_proc.stdout:
                        output = self._server_proc.stdout.read()
                except Exception:
                    pass
                raise RuntimeError(f"NV-CLIP server exited early. Logs:\n{output}")
            try:
                r = requests.get(f"{base_url}/v1/models", timeout=1)
                if r.ok:
                    ready = True
                    break
            except Exception:
                time.sleep(1)

        if not ready:
            output = ""
            try:
                if self._server_proc and self._server_proc.stdout:
                    output = "".join(self._server_proc.stdout.readlines()[-200:])
            except Exception:
                pass
            raise RuntimeError(f"NV-CLIP server did not become ready on {base_url}. Last logs:\n{output}")

        logger.info("NV-CLIP NIM is ready at %s/v1", base_url)
        # Persist for later calls
        self.base_url = base_url
        # Default to the concrete NV-CLIP variant used by the local server
        self.model_name = self.configuration.get("nim_model_name", "nvidia/nvclip-vit-h-14")
        return True

    def prepare_item_func(self, item: dl.Item):
        return item

    def embed(self, batch, **kwargs):
        image_batch = [Image.fromarray(item.download(save_locally=False, to_array=True)) for item in batch if 'image/' in item.mimetype]
        text_batch = [item.download(save_locally=False).read().decode() for item in batch if 'text/' in item.mimetype]
        image_indicies = [i for i, item in enumerate(batch) if 'image/' in item.mimetype]
        text_indicies = [i for i, item in enumerate(batch) if 'text/' in item.mimetype]
        # Convert images to base64 data URIs accepted by the endpoint
        encoded_images = []
        for img in image_batch:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            encoded_images.append(f"data:image/png;base64,{b64}")
        payload = {
            "input": text_batch + encoded_images,
            "model": getattr(self, "model_name", self.configuration.get("nim_model_name", "nvidia/nvclip-vit-h-14")),
            "encoding_format": "float",
        }
        response = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
        response.raise_for_status()
        data = response.json().get('data', [])
        outputs = [row.get('embedding') for row in data]

        # Map outputs back to original batch positions
        result = [None] * len(batch)
        t_count = len(text_batch)
        i_count = len(encoded_images)

        # Assign text embeddings
        for j, idx in enumerate(text_indicies):
            if j < t_count:
                result[idx] = outputs[j]

        # Assign image embeddings (come after texts)
        for k, idx in enumerate(image_indicies):
            pos = t_count + k
            if pos < len(outputs):
                result[idx] = outputs[pos]

        return result