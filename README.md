##
## dwarf_bench
##

“Dwarf bench” is a collection of patterns that attempt to capture performance characteristics of analytical queries. The idea is to extend the taxonomy of computational patterns defined in the article “The Landscape of Parallel Computing Research” (https://people.eecs.berkeley.edu/~krste/papers/BerkeleyView.pdf) published in 2006 to data analytics in heterogeneous environments. Implementing basic structures and algorithms once for multiple devices strives to find a balance between performance and specific capabilities usage, and implementation effort. We chose platform-agnostic tools to express our language of patterns (OpenCL, SYCL).

##
## Build
##

0. Requirements (see below): boost1.61, oclhelpers, opencl 1.2 (tested with nvidia 11.3, intel gfx & intel opencl cpu runtimes)
1. Get latest release of opencl helpers from https://github.com/kurapov-peter/oclhelpers/releases
2. Put the lib into your CMAKE_PREFIX_PATH or set oclhelpers_DIR env var
3. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
4. Install [CPU runtime](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html) (You'll only need the runtime: `sudo apt install intel-oneapi-runtime-opencl`)
5. Install [Intel gfx drivers](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html)
6. In order to run dpcpp tests with cpu, gpu and cuda follow these steps:
    - [Build dpcpp compiler with cuda support](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda)
    - Install [onedpl](https://github.com/oneapi-src/oneDPL/) (i.e. along with the [basekit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html#gs.24lvfe))
7. mkdir build && cd build && CXX=clang++ oclhelpers_DIR=/path/to/helpers cmake -DENABLE_DPCPP=ON .. && make -j`nproc`

##
## Docker - Running the dwarf CPU/GPU benchmark on a docker container.
##
Running the dwarf bench on a container requires the following steps:

And to make things easy we automated the process using the scripts
- DWARF_bench_automated.sh

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name  : DWARF_bench_automated.sh
# Purpose       : This script is written to use dwarf_bench to benchmark GPU/CPU
#                 It combines the work done by Petr A Kurapov on how to create a
#                 container for this purpose of benchmark and the work of
#                 Laurent Montigny on running dwarf_bench to benchmark GPU/CPU.
#                 This script automates the entire process that is:
# - Cleanup any previous container from previous run
# - git clone the repository create by Petr for this purpose: https://github.com/kurapov-peter/dwarf_bench
# - Create a docker image base on the Dockerfile
# - Create a container with that image
# - Then set some env params in the container, build the dwarf_bench binary
# - Once the build is done, the command dwarf_bench can be used to benchmark the
#   CPU and GPU. It's where the script benchmarks.sh written by Laurent is called
#   from DWARF_bench_run_in_the_container.sh script to do the job.
# Author        : Mamadou Diatta - mamadou.diatta@intel.com
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- DWARF_bench_run_in_the_container.sh

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name  : DWARF_bench_run_in_the_container.sh
# Purpose       : This script is written to use dwarf_bench to benchmark GPU/CPU
#                 It combines the build of the dwarf_bench binary inside
#                 the container and running the benchmark of GPU/CPU using the
#                 script benckmarks.sh writen by Laurent Montigny by setting up
#		  some necessary environment variables before hand.
# Author        : Mamadou Diatta - mamadou.diatta@intel.com
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- benchmarks.sh

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name  : benchmarks.sh
# Purpose       : This script is written to use dwarf_bench to benchmark GPU/CPU
#                 This is the work of Laurent Montigny to benchmark CPU/GPU, I
#                 took it and put some garde-fou aaround it and use it to run
#                 benchmark inside the container once the container is created.
#                 As you can see this script will create a lot of csv files with
#                 the results of the benchmark which content will look like this
# -------------------------------------------------------
#device_type,buf_size_bytes,host_time_ms,kernel_time_ms
#CPU,102400,1.054,0
#CPU,102400,0.017,0
#CPU,102400,0.014,0
#CPU,102400,0.014,0
# -------------------------------------------------------
# Author        : Mamadou Diatta - mamadou.diatta@intel.com
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

How can you then deploy this?
1) It must be on a server on which docker engine is install and running the
command [docker --version] should give you something like this output below
[Docker version 20.10.21, build baeda1f] if not then install docker engine.

2) Just get the following scripts into your current directory:

- DWARF_bench_automated.sh
- DWARF_bench_run_in_the_container.sh
- benchmarks.sh (Note: You can update this script to satisfy your GPU/CPU benchmarks needs)

3) You as a user must have SUDO access, if not this will not work. This is necesssary because you will be running some command that request root access like creating group and adding yourself to the groups.

4) Execute the only main script DWARF_bench_automated.sh which in turn will
the right script at the right time to produce at the end the necessary .csv files
at the end.

NOTE: At the start of the container, we mounted as a volume the dwarf_bench
directory inside a container as a volume. The dwarf_bench directory is create
by the git clone, so do not worry about it. And when the scripts finish their
job, your have on it the .csv files create by the benchmarks.sh. Those are
the files you need at the end.

Please read each script to get an idea about what is really happening if you
want the deep details.

If you want to learn about the bench_warf binary, you can once the container
is created run the following command to login [docker container exec -it spicy bash]
Once inside the container run [ dwarf_bench --help ] which output is as below

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DWARF_BENCH_ROOT is set to /build
You can change that with 'export DWARF_BENCH_ROOT=/your/path'
Dwarf bench:
  --help                Show help message
  --dwarf arg           Dwarf to run. List all with 'list' option.
  --input_size arg      Data array size, ususally a column size in elements
  --iterations arg      Number of iterations to run a bmark.
  --device arg          Device to run on.
  --report_path arg     Full/Relative path to a report file.
  --groups_count arg    Number of unique keys for dwarfs with keys (groupby,
                        hash build etc.).
  --executors arg       Number of executors for GroupByLocal.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

or you can run [ dwarf_bench list ] which output is as below

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DWARF_BENCH_ROOT is set to /build
You can change that with 'export DWARF_BENCH_ROOT=/your/path'
Supported dwarfs:
        ConstantExample
        ConstantExampleCAPI
        ConstantExampleDPCPP
        CuckooHashBuild
        DPLScan
        GroupBy
        GroupByLocal
        HashBuild
        HashBuildNonBitmask
        Join
        NestedLoopJoin
        Radix
        ReduceDPCPP
        TBBSort
        TwoPassScan
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --------------------------- end of READM.md ---------------------------------
# #############################################################################
