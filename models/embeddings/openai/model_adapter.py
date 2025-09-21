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
import select

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    """Embeddings adapter that boots the NV-CLIP NIM server inside the container.

    In some execution environments the Docker ENTRYPOINT is not invoked. This
    adapter explicitly starts the NIM server (same command as the image entrypoint)
    and waits until the OpenAI-compatible endpoint is ready on port 8000.
    """

    _server_proc: Optional[subprocess.Popen] = None

    def load(self, local_path, **kwargs):
        # Start the NIM server using Python 3.12 to ensure correct env/packages
        cmd = [
            "/usr/bin/python3.12",
            "-u",
            "-c",
            (
                "import re, sys; from nimlib.start_server import main; "
                "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0]); "
                "sys.exit(main())"
            ),
        ]
        # Enforce system Python 3.10 for any python invocations inside the script
        NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

        runtime_env = {**os.environ, "NGC_API_KEY": NVIDIA_API_KEY, "NVIDIA_API_KEY": NVIDIA_API_KEY, "ACCEPT_EULA": os.environ.get("ACCEPT_EULA", "Y")}
        runtime_env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        # Extra diagnostics and backtraces for better error reporting
        runtime_env.setdefault("NIM_LOG_LEVEL", "debug")
        runtime_env.setdefault("RUST_BACKTRACE", "1")
        # Pass-through optional org/team/cache settings if provided
        for var in ("NGC_ORG", "NGC_TEAM", "NIM_CACHE_ROOT"):
            val = os.environ.get(var)
            if val:
                runtime_env[var] = val
        # Ensure working directory is /opt/nim as some assets/configs are relative
        workdir = "/opt/nim"
        if not os.path.isdir(workdir):
            raise RuntimeError(f"Required workdir not found: {workdir}")

        print(f"Starting NV-CLIP NIM server via Python code: {cmd[0]} (cwd={workdir})")
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=runtime_env,
            cwd=workdir,
        )

        # Wait for readiness on the OpenAI-compatible endpoint; stream logs while waiting
        base_url = "http://127.0.0.1:8000"
        ready = False
        log_path = "/tmp/nim_startup.log"
        try:
            with open(log_path, "a") as _log:
                start_time = time.time()
                timeout_sec = int(os.environ.get("NIM_STARTUP_TIMEOUT_SEC", "5400"))  # default 90m
                i = 0
                while time.time() - start_time < timeout_sec:
                    # Stream any available output from the process
                    try:
                        readable, _, _ = select.select([self._server_proc.stdout, self._server_proc.stderr], [], [], 0.1)
                        for f in readable:
                            line = f.readline()
                            if line:
                                output = f"[nim] {line.strip()}"
                                print(output)
                                _log.write(output + "\n")
                                _log.flush()
                    except Exception:
                        pass

                    # If the process died early, surface last logs path
                    if self._server_proc.poll() is not None:
                        # Drain remaining output to log
                        try:
                            rem_out = self._server_proc.stdout.read() or ""
                            rem_err = self._server_proc.stderr.read() or ""
                            for line in rem_out.splitlines() + rem_err.splitlines():
                                if line:
                                    msg = f"[nim] {line}"
                                    print(msg)
                                    _log.write(msg + "\n")
                            _log.flush()
                        except Exception:
                            pass
                        rc = self._server_proc.returncode
                        raise RuntimeError(f"NV-CLIP server exited early (code {rc}). See log at {log_path}")

                    # Probe readiness every ~5s (prefer health/ready, fallback to models)
                    if i % 50 == 0:
                        try:
                            elapsed = int(time.time() - start_time)
                            for path in ("health/ready", "models"):
                                print(f"Checking readiness at {base_url}/v1/{path} (elapsed {elapsed}s)")
                                r = requests.get(f"{base_url}/v1/{path}", timeout=2)
                                if r.ok:
                                    ready = True
                                    break
                            if ready:
                                break
                        except Exception:
                            pass
                    i += 1
                    time.sleep(0.1)
        finally:
            pass

        if not ready:
            raise RuntimeError(f"NV-CLIP server did not become ready on {base_url}. See log at {log_path}")

        print(f"NV-CLIP NIM is ready at {base_url}/v1")
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


if __name__ == "__main__":
    adapter = ModelAdapter()
    adapter.load(None)
    adapter.embed(["Hello, world!"])
    print(adapter.embed(["Hello, world!"]))
