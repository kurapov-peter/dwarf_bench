# 100kib, 1mb, 2mb, 4mb, ... 512mb
# integrated gpu + cpu
SYCL_DEVICE_FILTER=cuda ./dwarf_bench DPLScanCuda --device=gpu --input_size=256 512 1024 2048 4096 8192 16384 32768 65536 --report_path="report_dpl_scan_small.csv" --iterations=9
