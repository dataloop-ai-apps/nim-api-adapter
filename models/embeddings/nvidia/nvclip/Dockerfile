FROM nvcr.io/nim/nvidia/nvclip:2.0.0

USER root
RUN apt update && apt install -y python3-dev python3-pip curl git

COPY .cache /opt/nim/.cache
RUN chown -R 1000:1000 /opt/nim/.cache

USER 1000

ENV HOME=/tmp

# install dtlpy into system Python despite PEP 668
RUN pip3 install --break-system-packages \
    https://storage.googleapis.com/dtlpy/dev/dtlpy-1.92.15-py3-none-any.whl --upgrade

# install extra packages into system Python (not user dir)
RUN pip3 install --break-system-packages \
    openai prometheus-client==0.20.0

ENTRYPOINT []