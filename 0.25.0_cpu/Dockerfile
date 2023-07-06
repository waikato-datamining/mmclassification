ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install MMCV
RUN pip install --no-cache-dir mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install MMClassification
RUN git clone https://github.com/open-mmlab/mmclassification.git /mmclassification && \
    cd /mmclassification && \
    git checkout tags/v0.25.0
WORKDIR /mmclassification
RUN pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir openmim==0.1.6 && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.2" && \
    pip install --no-cache-dir onnx==1.10.2 onnxruntime==1.10.0

ENV PYTHONPATH /mmclassification
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY mmcls_* /usr/bin/
COPY __init__.py /mmclassification/mmcls/datasets
COPY external.py /mmclassification/mmcls/datasets
COPY predict_common.py /mmclassification/tools
COPY predict_poll.py /mmclassification/tools
COPY predict_redis.py /mmclassification/tools
COPY train.py /mmclassification/tools
