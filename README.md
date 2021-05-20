# dwarf_bench

## build
0. Requirements (see below): boost1.61, oclhelpers, opencl 1.2 (tested with nvidia 11.3, intel gfx & intel opencl cpu runtimes)
1. Get latest release of opencl helpers from https://github.com/kurapov-peter/oclhelpers/releases
2. Put the lib into your CMAKE_PREFIX_PATH or set oclhelpers_DIR env var
3. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
4. Install [CPU runtime](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html) (You'll only need the runtime: `sudo apt install intel-oneapi-runtime-opencl`)
5. Install [Intel gfx drivers](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html)
5. mkdir build && cd build && cmake .. && make -j`nproc`