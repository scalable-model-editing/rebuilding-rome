FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /workspace/rebuilding-rome

COPY ./ .

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages sudo \
    build-essential libsystemd0 libsystemd-dev libudev0 libudev-dev cmake libncurses5-dev libncursesw5-dev git libdrm-dev \
    python3 python3-pip python3-setuptools

RUN pip install -r requirements.txt

# force re-install CUDA requirements to be compatible with 11.8
RUN pip install -U --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu118

# git lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install -y git-lfs
RUN git lfs pull

WORKDIR /workspace/rebuilding-rome
