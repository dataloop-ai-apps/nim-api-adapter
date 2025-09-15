FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root

RUN apt update && apt install -y python3-dev python3-pip curl git sudo docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Give newuser sudo privileges
RUN useradd -m newuser && \
    echo "newuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/newuser && \
    chmod 0440 /etc/sudoers.d/newuser

USER newuser

# Set environment variables
ENV HOME=/tmp

ENV NGC_API_KEY=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

# Install Dataloop and related packages
# RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.115.23-py3-none-any.whl
# RUN pip3 install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.115.23.100-py3-none-any.whl --user
# RUN pip3 install openai prometheus-client==0.20.0 --user

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
