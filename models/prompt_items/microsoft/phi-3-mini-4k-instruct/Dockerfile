FROM nvcr.io/nim/microsoft/phi-3-mini-4k-instruct:latest

USER root

RUN apt update && apt install -y python3-dev python3-pip curl git sudo docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



RUN usermod -aG sudo $(getent passwd 1000 | cut -d: -f1) && \
    echo "$(getent passwd 1000 | cut -d: -f1) ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/1000 && \
    chmod 0440 /etc/sudoers.d/1000

RUN chown -R 1000:1000 /opt/nim/llm/.venv

USER 1000


RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN /opt/nim/llm/.venv/bin/python /tmp/get-pip.py

ENV HOME=/tmp

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

RUN /opt/nim/llm/.venv/bin/python -m pip install dtlpy==1.105.6 --upgrade
RUN pip install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.105.6.100-py3-none-any.whl --upgrade
RUN /opt/nim/llm/.venv/bin/python -m pip install openai prometheus-client==0.20.0

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.15 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.15
