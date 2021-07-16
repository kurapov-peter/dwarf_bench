# Copyright (c) 2019-2020 Intel Corporation.
# SPDX-License-Identifier: BSD-3-Clause

# requires os-tools image
ARG base_image="intel/oneapi:os-tools-ubuntu18.04"
FROM "$base_image"

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# install Intel(R) oneAPI Runtime Libraries
RUN apt-get update -y && \
apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
intel-oneapi-runtime-libs \
intel-oneapi-runtime-opencl \
intel-oneapi-runtime-tbb \
--

# install Intel GPU drivers
RUN echo 'deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main' \
> /etc/apt/sources.list.d/intel-graphics.list

ARG url=https://repositories.intel.com/graphics/intel-graphics.key
ADD $url /
RUN file=$(basename "$url") && \
    apt-key add "$file" && \
    rm "$file"

RUN apt-get update -y && \
apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
intel-opencl \
intel-level-zero-gpu \
level-zero \
level-zero-devel

#Install dwarf_bench
RUN apt-get install -y git
RUN apt-get install -y wget
RUN cd home && git clone https://github.com/kurapov-peter/dwarf_bench

#Install oclhelpers
RUN cd home && mkdir oclhelpers \
    && cd oclhelpers \
    &&wget https://github.com/kurapov-peter/oclhelpers/releases/download/v0.1.3/oclhelpers-v0.1.3-Release.tar.gz \
    && tar -xzvf oclhelpers-v0.1.3-Release.tar.gz

#Install OpenCL
RUN apt-get install -y ocl-icd-opencl-dev && apt-get install -y opencl-headers

#Set oclheaders_Dir
ENV oclhelpers_DIR=/home/oclhelpers/lib/cmake/oclhelpers

#Install ninja for clang++
RUN apt-get install -y ninja-build

#Install python3.9 for clang++
RUN apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.9

#Install clang++
RUN export DPCPP_HOME=/root/sycl_workspace \
    && mkdir $DPCPP_HOME \
    && cd $DPCPP_HOME \
    && git clone https://github.com/intel/llvm -b sycl \
    && python3.9 $DPCPP_HOME/llvm/buildbot/configure.py \
    && python3.9  $DPCPP_HOME/llvm/buildbot/compile.py

#Set vars for compiler annd sycl lib
ENV PATH=/root/sycl_workspace/llvm/build/bin:$PATH
ENV LD_LIBRARY_PATH=/root/sycl_workspace/llvm/build/lib:$LD_LIBRARY_PATH

#Install boost
RUN apt-get install -y libboost-all-dev

#Install oneDPL
RUN cd root && mkdir onepdl && cd onepdl \
    && wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17889/l_oneDPL_p_2021.4.0.337_offline.sh \
    && bash l_oneDPL_p_2021.4.0.337_offline.sh -s -a -s --eula accept
