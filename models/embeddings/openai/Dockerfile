FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root
# Install Python 3.10 from deadsnakes and make it default for all users
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      software-properties-common gnupg curl ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-distutils python3.10-dev build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    python3 -V && \
    python3 -m pip install --no-cache-dir --upgrade "pip<25" "setuptools<70" wheel && \
    python3 -m pip install --no-cache-dir \
      docopt==0.6.2 \
      openai \
      requests \
      pillow \
      dtlpy==1.114.17 \
      https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.114.17.100-py3-none-any.whl \
      psutil

# Drop privileges for runtime
USER 1000
ENV HOME=/tmp




# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.14 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.14
# docker run -it --gpus all gcr.io/viewo-g/piper/agent/runner/cpu/nvclip:0.1.10 bash

