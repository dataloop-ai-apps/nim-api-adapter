FROM nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest

USER root
RUN apt update && apt install -y python3-dev python3-pip curl git


USER 1000
ENV HOME=/tmp

RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.92.15-py3-none-any.whl --upgrade
RUN pip3 install openai prometheus-client==0.20.0 --user

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.7 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.7
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-11b-vision:0.1.7 bash