FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
ENV HOME=/tmp
WORKDIR /tmp

RUN git clone --recursive https://github.com/nvidia-riva/python-clients.git

RUN pip install grpcio-tools==1.67.1 && \
    cd /tmp/python-clients && \
    pip install --no-build-isolation .

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/riva_asr:0.0.1 -f nodes/asr/Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/riva_asr:0.0.1
