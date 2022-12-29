FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
ARG USER_NAME
ARG USER_ID
ARG PYTHON_VER="3.8"

# Install apt-packages
RUN sed -i "s/archive.ubuntu/mirror.kakao/g" /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    software-properties-common gcc sudo git curl unzip wget locales openssh-server \
    python${PYTHON_VER} python${PYTHON_VER}-distutils python${PYTHON_VER}-dev \
    neovim tree tmux screen \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists

# Install python-packages
RUN ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python3 \
 && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
 && python3 get-pip.py \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install \
    cython scikit-build click requests \
    tqdm pyspng ninja imageio-ffmpeg==0.4.3 \
    psutil scipy matplotlib opencv-python scikit-image pycocotools \
 && python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html \
 && python3 -m pip cache purge

# Create user
RUN useradd --create-home --shell /bin/bash --uid ${USER_ID} ${USER_NAME} \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && usermod -aG sudo ${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN touch /home/${USER_NAME}/.Xauthority
