FROM nvcr.io/nim/meta/llama-3.2-90b-vision-instruct

USER root
RUN apt update && apt install -y python3-dev python3-pip curl git


USER 1000
ENV HOME=/tmp

RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.92.15-py3-none-any.whl --upgrade
RUN pip3 install openai prometheus-client==0.20.0 --user

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-90b-vision:0.1.5 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-90b-vision:0.1.5
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/llama3-2-90b-vision:0.1.5 bash