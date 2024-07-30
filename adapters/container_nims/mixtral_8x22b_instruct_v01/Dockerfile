FROM nvcr.io/nim/mistralai/mixtral-8x22b-instruct-v01:latest

USER root
RUN apt update && apt install -y python3-dev python3-pip
RUN apt update && apt install -y curl

RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.92.15-py3-none-any.whl --upgrade
RUN pip3 install openai prometheus-client==0.20.0


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/mixtral822b:0.1.4 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/mixtral822b:0.1.4
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/mixtral822b:0.1.4 bash