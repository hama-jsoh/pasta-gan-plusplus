FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04 as base
FROM base as base-amd64
FROM base-amd64
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ARG USER_NAME
ARG USER_ID

# Install apt-packages
RUN sed -i "s/archive.ubuntu/mirror.kakao/g" /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y \
    gcc \
    sudo \
    tree \
    vim \
    openssh-server

# Install python-packages
RUN python3 -m pip install --upgrade pip \
    cython scikit-build click requests \
    tqdm pyspng ninja imageio-ffmpeg==0.4.3 \
    psutil scipy matplotlib opencv-python scikit-image pycocotools \
 && python3 -m pip install torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html \
 && python3 -m pip cache purge

# Create user
RUN useradd --create-home --shell /bin/bash --uid ${USER_ID} ${USER_NAME} \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && usermod -aG sudo ${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN touch /home/${USER_NAME}/.Xauthority
