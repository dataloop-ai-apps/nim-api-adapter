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
import threading
import random

logger = logging.getLogger("NIM Adapter")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

class ModelAdapter(dl.BaseModelAdapter):
    """Embeddings adapter that boots the NV-CLIP NIM server inside the container.

    In some execution environments the Docker ENTRYPOINT is not invoked. This
    adapter explicitly starts one or more NIM servers (same command as the image
    entrypoint) and waits until their OpenAI-compatible endpoints are ready on
    configurable ports.
    """

    _server_proc: Optional[subprocess.Popen] = None
    _log_thread: Optional[threading.Thread] = None
    # Multi-instance support
    _server_procs = []
    _log_threads = []

    def load(self, local_path, **kwargs):
        # Command used to start the NIM server (same as image entrypoint)
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

        NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
        num_servers = max(1, int(self.configuration.get("num_servers", 1)))
        base_port = int(self.configuration.get("nim_base_port", os.environ.get("NIM_HTTP_API_PORT", "8000")))

        # Base runtime environment shared by all instances
        base_env = {
            **os.environ,
            "NGC_API_KEY": NVIDIA_API_KEY,
            "NVIDIA_API_KEY": NVIDIA_API_KEY,
            "ACCEPT_EULA": os.environ.get("ACCEPT_EULA", "Y"),
            "NIM_FORCE_PYTHON_312": "1",
        }
        base_env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        base_env.setdefault("NIM_LOG_LEVEL", "warning")
        base_env.setdefault("RUST_BACKTRACE", "1")
        for var in ("NGC_ORG", "NGC_TEAM", "NIM_CACHE_ROOT", "NIM_MANIFEST_PROFILE"):
            val = os.environ.get(var)
            if val:
                base_env[var] = val

        # Ensure working directory is /opt/nim as some assets/configs are relative
        workdir = "/opt/nim"
        if not os.path.isdir(workdir):
            raise RuntimeError(f"Required workdir not found: {workdir}")

        self._server_procs = []
        self._log_threads = []
        self.base_urls = []

        # Start N instances on incrementing ports using NIM_HTTP_API_PORT
        # Ref: NIM_HTTP_API_PORT controls the service port inside the container.
        # See NVIDIA docs: https://docs.nvidia.com/nim/nvclip/latest/configuration.html
        for idx in range(num_servers):
            port = base_port + (10*idx)
            runtime_env = dict(base_env)
            runtime_env["NIM_HTTP_API_PORT"] = str(port)

            print(f"Starting NV-CLIP NIM instance {idx+1}/{num_servers} on port {port}: {cmd[0]} (cwd={workdir})")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=runtime_env,
                cwd=workdir,
            )
            self._server_procs.append(proc)

            # Per-instance log streamer
            log_path = f"/tmp/nim_startup_{port}.log"
            t = threading.Thread(
                target=self._stream_nim_logs,
                args=(idx,proc, log_path),
                daemon=True,
            )
            t.start()
            self._log_threads.append(t)
            self.base_urls.append(f"http://127.0.0.1:{port}")

        # Readiness checks for all instances
        start_time = time.time()
        timeout_sec = int(os.environ.get("NIM_STARTUP_TIMEOUT_SEC", "5400"))

        for idx, (proc, base_url) in enumerate(zip(self._server_procs, self.base_urls)):
            ready = False
            i = 0
            while time.time() - start_time < timeout_sec:
                if proc.poll() is not None:
                    rc = proc.returncode
                    raise RuntimeError(
                        f"NV-CLIP instance on {base_url} exited early (code {rc}). See per-instance log file."
                    )
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
            if not ready:
                raise RuntimeError(f"NV-CLIP instance did not become ready on {base_url}")

        self.model_name = self.configuration.get("nim_model_name", "nvidia/nvclip-vit-h-14")
        print(f"NV-CLIP NIM instances ready: {', '.join(url + '/v1' for url in self.base_urls)}")
        return True

    def _stream_nim_logs(self, idx: int, proc: subprocess.Popen, log_path: str) -> None:
        """Continuously stream NIM server logs to stdout and a file until process exits.

        Runs as a daemon thread. Safe to call while main thread performs readiness checks.
        """
        try:
            with open(log_path, "a") as _log:
                while True:
                    try:
                        if proc.poll() is not None:
                            # Process ended; drain any remaining output once and exit
                            for f in (proc.stdout, proc.stderr):
                                try:
                                    while True:
                                        line = f.readline()
                                        if not line:
                                            break
                                        output = f"[nim:{idx}] {line.strip()}"
                                        print(output)
                                        _log.write(output + "\n")
                                        _log.flush()
                                except Exception:
                                    pass
                            break

                        # Stream available output without blocking
                        try:
                            readable, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.2)
                            for f in readable:
                                line = f.readline()
                                if line:
                                    output = f"[nim:{idx}] {line.strip()}"
                                    print(output)
                                    _log.write(output + "\n")
                                    _log.flush()
                        except Exception:
                            pass
                        time.sleep(0.05)
                    except Exception:
                        time.sleep(0.1)
        except Exception:
            # Ensure logging thread never crashes the application
            pass

    def prepare_item_func(self, item: dl.Item):
        return item

    def embed(self, batch, **kwargs):
        logger.info(f"embed: received batch size={len(batch)}")
        image_batch = [Image.fromarray(item.download(save_locally=False, to_array=True)) for item in batch if 'image/' in item.mimetype]
        text_batch = [item.download(save_locally=False).read().decode() for item in batch if 'text/' in item.mimetype]
        image_indices = [i for i, item in enumerate(batch) if 'image/' in item.mimetype]
        text_indices = [i for i, item in enumerate(batch) if 'text/' in item.mimetype]
        logger.debug(f"embed: prepared inputs: texts={len(text_batch)}, images={len(image_batch)}")
        # Convert images to base64 data URIs accepted by the endpoint
        encoded_images = []
        for img in image_batch:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            encoded_images.append(f"data:image/png;base64,{b64}")
        model_name = getattr(self, "model_name", self.configuration.get("nim_model_name", "nvidia/nvclip-vit-h-14"))
        payload = {
            "input": text_batch + encoded_images,
            "model": model_name,
            "encoding_format": "float",
        }
        total_inputs = len(payload["input"])
        # Choose instance randomly per request; retry other instances in random order
        order = list(range(len(self.base_urls)))
        random.shuffle(order)
        last_exc = None
        response = None
        for attempt_idx, idx in enumerate(order, start=1):
            target_url = self.base_urls[idx]
            logger.debug(
                f"embed: POST {target_url}/v1/embeddings model={model_name} "
                f"total_inputs={total_inputs} (texts={len(text_batch)}, images={len(encoded_images)}) attempt={attempt_idx}/{len(self.base_urls)}"
            )
            _t0 = time.perf_counter()
            try:
                response = requests.post(f"{target_url}/v1/embeddings", json=payload)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                logger.warning(f"embed: request failed on {target_url} ({e}); trying next instance")
                continue
        if response is None:
            logger.exception("embed: all instances failed to serve the request")
            raise last_exc if last_exc else RuntimeError("All NIM instances failed")
        _dt_ms = (time.perf_counter() - _t0) * 1000.0
        logger.info(f"embed: request ok status={getattr(response, 'status_code', -1)} duration_ms={_dt_ms:.1f} inputs={total_inputs}")
        data = response.json().get('data', [])
        outputs = [row.get('embedding') for row in data]
        logger.debug(f"embed: received {len(outputs)} embeddings from service")

        # Map outputs back to original batch positions
        result = [None] * len(batch)
        t_count = len(text_batch)

        # Assign text embeddings
        for j, idx in enumerate(text_indices):
            if j < t_count:
                result[idx] = outputs[j]

        # Assign image embeddings (come after texts)
        for k, idx in enumerate(image_indices):
            pos = t_count + k
            if pos < len(outputs):
                result[idx] = outputs[pos]

        num_missing = sum(1 for r in result if r is None)
        if num_missing:
            logger.warning(
                f"embed: {num_missing}/{len(batch)} results missing after mapping " f"(texts={t_count}, images={len(image_batch)}, outputs={len(outputs)})"
            )
        else:
            logger.debug(f"embed: successfully mapped embeddings for all {len(batch)} inputs")

        return result


if __name__ == "__main__":
    adapter = ModelAdapter()
    adapter.load(None)
    adapter.embed(["Hello, world!"])
    print(adapter.embed(["Hello, world!"]))
