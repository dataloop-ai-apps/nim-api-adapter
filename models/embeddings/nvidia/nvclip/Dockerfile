FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root

RUN apt update && apt install -y \
    python3-dev python3-pip python3-venv curl git sudo docker.io \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m newuser && \
    echo "newuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/newuser && \
    chmod 0440 /etc/sudoers.d/newuser

# create a private venv (avoid PEP 668), then hand it to newuser
RUN python3 -m venv /opt/venv && chown -R newuser:newuser /opt/venv

USER newuser
ENV HOME=/tmp
ENV PATH="/opt/venv/bin:$PATH"

# install deps into our venv (no --user, no get-pip)
RUN python -m pip install --upgrade pip \
 && pip install --upgrade dtlpy \
 && pip install --upgrade https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.105.6.100-py3-none-any.whl \
 && pip install openai prometheus-client==0.20.0

ENTRYPOINT []
