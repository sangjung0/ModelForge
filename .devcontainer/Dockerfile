# Build environment
FROM ubuntu:22.04 AS builder

# Install Build tools
RUN apt-get update && apt-get install -y \
  wget \
  curl \
  git \
  build-essential \
  ca-certificates \
  cmake

# Install CUDA 12.6
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
  && wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb

# Install cuDNN 9.6
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb

# Install dlib 
RUN git clone https://github.com/davisking/dlib /tmp/dlib \
  && mkdir /tmp/dlib/build \
  && cd /tmp/dlib/build \
  && cmake .. \
  && cmake --build . --config Release

# Runtime environment
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y \
  python3.11 \
  python3-pip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  busybox \
  libgtk2.0-dev \
  pkg-config \
  qtbase5-dev \
  qt5-qmake \
  cmake \
  libxkbcommon-x11-0 \
  build-essential \
  libboost-python-dev \
  libboost-system-dev \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.6
COPY --from=builder /cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 
COPY --from=builder /cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb /
RUN dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb \
  && rm -rf cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb \
  && cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update && apt-get -y install cuda-drivers cuda-toolkit-12-6 


# Install cuDNN 9.6
COPY --from=builder /cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb /
RUN dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb \
  && rm -rf cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb \
  && cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update && apt-get -y install cudnn cudnn-cuda-12

# Install dlib
COPY --from=builder /tmp/dlib /tmp/dlib
RUN cd /tmp/dlib && python3 setup.py install && rm -rf /tmp/dlib

# Install pip requirements
COPY .devcontainer/requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir -r requirements.txt

# ENV
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"


# WORKDIR /app
# COPY . /app

# Xserver
ENV DISPLAY host.docker.internal:0.0

#CMD ["python", "main.py"]