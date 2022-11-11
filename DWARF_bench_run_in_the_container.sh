#!/bin/bash -x
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name  : benchmarks.sh
# Purpose       : This script is written to use dwarf_bench to benchmark GPU/CPU
#		  This script combine the build of the dwarf_bench binary inside
#		  the container and running the benchmark of GPU/CPU using the 
#		  script benckmarks.sh writen by Laurent Montigny
# Author        : Mamadou Diatta - mamadou.diatta@intel.com
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Let us set some env variable to allow the build to proceed
export https_proxy=http://proxy-dmz.intel.com:912
export http_proxy=http://proxy-dmz.intel.com:912
mkdir build && cd build
CXX=dpcpp cmake /dwarf_bench/dwarf_bench/ -DENABLE_DPCPP=on  -DCMAKE_BUILD_TYPE=Release
#CXX=dpcpp cmake /dwarf_bench/dwarf_bench/ -DENABLE_DPCPP=on -DENABLE_TESTS=on
cmake --build . --parallel 4
# The binary of dwarf_bench is created and needed to be move to the right dir
cp ../dwarf_bench/benchmarks.sh .
#cp ../dwarf_bench/benchmarks.sh.cp .
# Make sure it is executable 
chmod 755 ./benchmarks.sh
#chmod 755 ./benchmarks.sh.cp
# Execute the bencmarks.sh script which calls the dwarf_bench binary
./benchmarks.sh | tee ./benchmarks_output.$$.txt
#./benchmarks.sh.cp | tee ./benchmarks_.cpoutput.$$.txt
# Then move the all CVS files to the right place
pwd
ls -al *csv
mv *csv ../dwarf_bench/
ls -al ../dwarf_bench/*.csv
