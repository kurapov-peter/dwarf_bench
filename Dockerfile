FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    sudo \
    ca-certificates \
    gnupg2 \
    --

RUN update-ca-certificates

RUN echo 'deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main' \
    > /etc/apt/sources.list.d/intel-graphics.list
ARG url=https://repositories.intel.com/graphics/intel-graphics.key
ADD $url /
RUN file=$(basename "$url") && \
    apt-key add "$file" && \
    rm "$file"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    --

RUN apt-get install -y wget

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | apt-key add -
RUN echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
    intel-oneapi-tbb \
    --

ARG GIT_SSL_NO_VERIFY=1

ARG oclhelpers_version=0.1.3

RUN mkdir /oclhelpers \
    && cd /oclhelpers \
    && wget https://github.com/kurapov-peter/oclhelpers/releases/download/v${oclhelpers_version}/oclhelpers-v${oclhelpers_version}-Release.tar.gz \
    && tar -xzvf oclhelpers-v${oclhelpers_version}-Release.tar.gz

#Set oclheaders_Dir
ENV oclhelpers_DIR=/oclhelpers/lib/cmake/

RUN apt-get install -y ninja-build libboost-all-dev

ENV DPCPP_HOME="/dpcpp"
RUN mkdir ${DPCPP_HOME} && cd ${DPCPP_HOME} 

ARG sycl_llvm_sha="f71a1d5c6088c82f2ec0aa1d8b88c19db227d802"
RUN cd ${DPCPP_HOME} && git clone https://github.com/intel/llvm -b sycl
#RUN cd ${DPCPP_HOME} && git checkout ${sycl_llvm_sha}
RUN python3 ${DPCPP_HOME}/llvm/buildbot/configure.py
RUN python3 ${DPCPP_HOME}/llvm/buildbot/compile.py

ENV PATH="${DPCPP_HOME}/llvm/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="${DPCPP_HOME}/llvm/build/lib:${LD_LIBRARY_PATH}"

RUN apt-get install -y intel-oneapi-runtime-opencl opencl-headers
RUN apt-get install -y intel-oneapi-runtime-tbb
RUN apt-get install -y intel-oneapi-tbb-devel
RUN apt-get install -y libtbb-dev

RUN mkdir /onepdl && cd /onepdl \
    && wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17889/l_oneDPL_p_2021.4.0.337_offline.sh \
    && bash l_oneDPL_p_2021.4.0.337_offline.sh -s -a -s --eula accept
