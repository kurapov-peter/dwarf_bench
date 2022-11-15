#!/bin/bash
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Program name  : DWARF_bench_run_in_the_container.sh
# Purpose       : This script is written to use dwarf_bench to benchmark GPU/CPU
#                 This is the work of Laurent Montigny to benchmark CPU/GPU, I
#                 took it and put some garde-fou aaround it and use it to run
#                 benchmark inside the container once the it is created.
#		  As you can see this script will create a lot of csv files with
#		  the results of the benchmark which content will look like this
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


rm -rf report*.csv

sycl-ls | grep ext_oneapi_level_zero| grep -v grep
if [ $? == 0 ]
then
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench Radix --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_radix.csv" --iterations=9
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench GroupBy --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_GoupBy_GPU.csv" --iterations=9
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench Radix --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_RadixSort_GPU.csv" --iterations=9
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench TBBSort --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_TBBSort_GPU.csv" --iterations=4
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench Join --device=gpu --input_size=25600 262144 524288 1048576 --report_path="report_join_GPU.csv" --iterations=9
	SYCL_DEVICE_FILTER=opencl:gpu:2 ./dwarf_bench DPLScan --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_DPLScan_GPU.csv" --iterations=9
else
	echo
	echo "In this container there is no gpu!"
	echo
fi


sycl-ls | egrep "host:host:0" | grep -v grep
if [ $? == 0 ]
then
	./dwarf_bench Radix --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_radix_CPU.csv" --iterations=9
	./dwarf_bench GroupBy --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_GroupBy_CPU.csv" --iterations=9
	./dwarf_bench Radix --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_Radixsort_CPU.csv" --iterations=9
	./dwarf_bench TBBSort --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_TBBSort_CPU.csv" --iterations=4
#	./dwarf_bench TBBSort --device=cpu --input_size=25600 --report_path="report_TBBSort_CPU.csv" --iterations=4
	./dwarf_bench Join --device=cpu --input_size=25600 262144 524288 1048576 --report_path="report_join_CPU.csv" --iterations=9
	./dwarf_bench DPLScan --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 --report_path="report_DPLScan_CPU.csv" --iterations=9
else
	echo
	echo "This container is very confused about its cpu. Sonething wrong with the cycle"
	echo
fi
