# 100kib, 1mb, 2mb, 4mb, ... 512mb
# nvidia gpu
SYCL_DEVICE_FILTER=cuda ./dwarf_bench DPLScanCuda --device=gpu --input_size=25600 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 --report_path="report_dpl_scan.csv" --iterations=9
