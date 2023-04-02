FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Image parameters
ARG DEBIAN_FRONTEND=noninteractive
ARG USER=docker
ARG PW=docker
ARG UID=1000
ARG GID=1000

# Basic Setup
RUN apt-get update \
    && apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 
ENV LC_ALL en_US.UTF-8

# Install dev base
RUN apt-get update -y && apt-get install -y \
    software-properties-common  \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    git \
    unzip \
    wget \
    vim \
    htop \
    bash-completion \
    python3.10 \
    python3-pip \
    python-is-python3 \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install pip --upgrade && \
    pip3 install -r requirements.txt && \
    rm requirements.txt

# Install Detectron2 
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# ~ Setup user
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd

USER ${UID}:${GID}
WORKDIR /home/${USER}
