# 100kib, 1mb, 2mb, 4mb, ... 512mb
# integrated gpu + cpu
./dwarf_bench DPLScan --device=igpu --input_size=256 512 1024 2048 4096 8192 16384 32768 65536 --report_path="report_dpl_scan_small.csv" --iterations=9
./dwarf_bench DPLScan --device=cpu --input_size=256 512 1024 2048 4096 8192 16384 32768 65536 --report_path="report_dpl_scan_small.csv" --iterations=9
