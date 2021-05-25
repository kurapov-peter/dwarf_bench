# 100kib, 1mb, 2mb, 4mb, ... 512mb
# integrated gpu + cpu
./dwarf_bench Radix --device=igpu --input_size=25600 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 --report_path="report_radix.csv" --iterations=9
./dwarf_bench Radix --device=cpu --input_size=25600 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 --report_path="report_radix.csv" --iterations=9
