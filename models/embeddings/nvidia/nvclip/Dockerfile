FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root

# Install system dependencies
RUN apt update && apt install -y python3-dev python3-pip curl git docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# No sudo access needed - subprocess commands removed from model adapter

# Set ownership for nim cache directory
RUN chown -R 1000:1000 /opt/nim/.cache

USER 1000

# Set environment variables
ENV HOME=/tmp
ENV LOCAL_NIM_CACHE=/opt/nim/.cache

# CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Install Python dependencies
RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN /opt/nim/llm/.venv/bin/python /tmp/get-pip.py

# Install Dataloop and related packages
RUN /opt/nim/llm/.venv/bin/python -m pip install dtlpy==1.105.6 --upgrade
RUN pip install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.105.6.100-py3-none-any.whl --upgrade
RUN /opt/nim/llm/.venv/bin/python -m pip install openai prometheus-client==0.20.0

# Create cache directory
RUN mkdir -p "$LOCAL_NIM_CACHE"

# Expose port 8000
EXPOSE 8000

# Set working directory
WORKDIR /opt/nim

# Default command (can be overridden)
CMD ["python", "-m", "http.server", "8000"]

# Build and run instructions:
# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/nvclip:2.0.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/nvclip:2.0.0
# 
# To run with the provided command:
# export NGC_API_KEY=<PASTE_API_KEY_HERE>f
# export LOCAL_NIM_CACHE=~/.cache/nim
# mkdir -p "$LOCAL_NIM_CACHE"
# docker run -it --rm \
#     --gpus all \
#     -e NGC_API_KEY=$NGC_API_KEY \
#     -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
#     -p 8000:8000 \
#     gcr.io/viewo-g/piper/agent/runner/gpu/nvclip:2.0.0
