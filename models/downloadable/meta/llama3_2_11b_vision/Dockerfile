FROM nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest

USER root

RUN apt update && apt install -y python3-dev python3-pip curl git sudo docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Give newuser sudo privileges
RUN useradd -m newuser && \
    echo "newuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/newuser && \
    chmod 0440 /etc/sudoers.d/newuser

RUN chown -R newuser:newuser /opt/nim/llm/.venv

USER newuser


RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN /opt/nim/llm/.venv/bin/python /tmp/get-pip.py

ENV HOME=/tmp

RUN /opt/nim/llm/.venv/bin/python -m pip install dtlpy --upgrade
RUN pip install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.105.6.100-py3-none-any.whl --upgrade
RUN /opt/nim/llm/.venv/bin/python -m pip install openai prometheus-client==0.20.0

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.15 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.15
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.15 bash