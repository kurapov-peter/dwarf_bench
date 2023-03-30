FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
    build-essential \
    cmake \
    git \
    sudo \
    wget \
    ninja-build \
    python3.9 \
    libboost-all-dev \
    libtbb-dev \
    g++ \
    gcc \
    libstdc++-12-dev \
    --

ARG compute_runtime_version="22.49.25018.24"
ARG igc_version="1.0.12812.24"
ARG level_zero_version="1.9.4"
RUN mkdir /intel-gpu-drivers && cd /intel-gpu-drivers && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v${level_zero_version}/level-zero-devel_${level_zero_version}+u18.04_amd64.deb && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v${level_zero_version}/level-zero_${level_zero_version}+u18.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-${igc_version}/intel-igc-core_${igc_version}_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-${igc_version}/intel-igc-opencl_${igc_version}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${compute_runtime_version}/intel-level-zero-gpu-dbgsym_1.3.25018.24_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/${compute_runtime_version}/intel-level-zero-gpu_1.3.25018.24_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${compute_runtime_version}/intel-opencl-icd-dbgsym_${compute_runtime_version}_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/${compute_runtime_version}/intel-opencl-icd_${compute_runtime_version}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${compute_runtime_version}/libigdgmm12_22.3.0_amd64.deb

RUN cd /intel-gpu-drivers && dpkg -i *.deb

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-0 cuda-drivers-525

ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

ENV DPCPP_HOME="/dpcpp"
RUN mkdir ${DPCPP_HOME} && cd ${DPCPP_HOME} 

ARG GIT_SSL_NO_VERIFY=1
ARG sycl_llvm_sha="f292b05de679b952e4188d0ce20c1ffad2c584ba"
RUN cd ${DPCPP_HOME} && git clone https://github.com/intel/llvm -b sycl
RUN cd ${DPCPP_HOME}/llvm && git checkout ${sycl_llvm_sha}
RUN CC=gcc CXX=g++ python3 ${DPCPP_HOME}/llvm/buildbot/configure.py --cuda
RUN CC=gcc CXX=g++ python3 ${DPCPP_HOME}/llvm/buildbot/compile.py

ENV PATH="${DPCPP_HOME}/llvm/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="${DPCPP_HOME}/llvm/build/lib:${LD_LIBRARY_PATH}"

RUN apt-get install -y ocl-icd-* opencl-headers intel-opencl-icd

RUN wget https://github.com/oneapi-src/oneDPL/archive/refs/tags/oneDPL-2021.7.1-release.tar.gz
RUN tar xf oneDPL-2021.7.1-release.tar.gz
RUN cd oneDPL-oneDPL-2021.7.1-release && mkdir build && cd build && cmake .. && cmake --build . --target install
