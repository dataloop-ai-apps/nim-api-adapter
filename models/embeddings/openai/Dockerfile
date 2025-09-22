FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root
# Install Python 3.10 from deadsnakes and make it default for all users
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      software-properties-common gnupg curl ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-distutils python3.10-dev build-essential \
      libffi-dev libssl-dev pkg-config rustc cargo && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    python3 -V && \
    python3 -m pip install --no-cache-dir --upgrade "pip<25" "setuptools<70" wheel && \
    python3 -m pip install --no-cache-dir \
      cffi \
      cryptography \
      docopt==0.6.2 \
      openai \
      requests \
      pillow \
      dtlpy==1.114.17 \
      https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.114.17.100-py3-none-any.whl \
      psutil

# Ensure Python 3.12 stdlib venv/distutils are present (best-effort)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12-venv python3.12-distutils || true

# Ensure Python 3.12 has pip/setuptools/wheel (distutils shim)
RUN /usr/bin/python3.12 -m ensurepip || true && \
    /usr/bin/python3.12 -m pip install --no-cache-dir --upgrade "pip<25" "setuptools<70" wheel packaging

# Ensure distutils-compat behavior for Python 3.12-based runtime stacks
ENV SETUPTOOLS_USE_DISTUTILS=local

# Optional hardening: always set the shim for python3.12 via sitecustomize
RUN python3.12 - <<'PY'
import pathlib
p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/sitecustomize.py')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text('import os; os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS","local")\n')
print('sitecustomize written to', p)
PY

# Persist and use a local cache for model artifacts
ENV NIM_CACHE_ROOT=/opt/nim/.cache
RUN mkdir -p ${NIM_CACHE_ROOT} && chown -R 1000:1000 ${NIM_CACHE_ROOT}

# Drop privileges for runtime
USER 1000
ENV HOME=/tmp

# Provide NGC key at build time to prefetch model weights into cache (do NOT hardcode)
ARG NGC_API_KEY
ENV NGC_API_KEY=${NGC_API_KEY}
ENV NVIDIA_API_KEY=${NGC_API_KEY}
ENV ACCEPT_EULA=Y

# Prefetch model artifacts into cache without binding the server
WORKDIR /opt/nim
RUN /usr/bin/python3.12 - <<'PY'
from nimlib.nimutils import get_model_manifest, download_models
mm = get_model_manifest()
download_models(mm)
print("Prefetch complete")
PY






# docker build --no-cache --build-arg NGC_API_KEY=46CFpoBSe*******OZSG9dW8kg -t gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.22 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.22
# docker run -it --gpus all gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.10 bash


# # To run the server locally inside the docker container
# docker run -it --rm --entrypoint /bin/bash gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.10
# docker run -it --gpus all --rm -v E:\Applications\nim-api-adapter\models\embeddings\openai:/opt/nim/app --entrypoint /bin/bash gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.20
