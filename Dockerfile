FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
ENV HOME=/tmp
RUN pip install openai


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/nim-api:0.1.11 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/nim-api:0.1.11

