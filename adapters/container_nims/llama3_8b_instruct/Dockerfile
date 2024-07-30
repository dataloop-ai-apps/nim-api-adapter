FROM nvcr.io/nim/meta/llama3-8b-instruct:1.0.0

USER root
RUN apt update && apt install -y python3-dev python3-pip
RUN apt update && apt install -y curl

RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.92.15-py3-none-any.whl --upgrade
RUN pip3 install openai prometheus-client==0.20.0

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/llama3-8b-instruct:0.1.2 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/llama3-8b-instruct:0.1.2
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/llama3-8b-instruct:0.1.2 bash