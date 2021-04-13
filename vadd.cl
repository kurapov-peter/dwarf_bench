void kernel vadd(global const int *src1, global const int *src2,
                 global int *restrict out) {
  int id = get_global_id(0);
  out[id] = src1[id] + src2[id];
}