bool lt_filter(int value, int condition) { return value < condition; }

void kernel simple_two_pass_scan(global const int *src, int src_size,
                                 global int *restrict out, global int *out_size,
                                 int filter_value, global int *restrict prefix,
                                 global int *debug) {
  int id = get_global_id(0);
  int tnum = get_global_size(0);

  // expects it to be greater
  int work_per_thread = src_size / tnum;

  int sz = 0;
  for (int i = id * work_per_thread; i < (id + 1) * work_per_thread; i++) {
    if (lt_filter(src[i], filter_value)) {
      sz++;
    }
  }
  prefix[id + 1] = sz;

  barrier(CLK_LOCAL_MEM_FENCE);
  // prefix sum todo
  if (id == 0) {
    prefix[0] = 0;
    debug[0] = prefix[0];
    for (int i = 1; i <= tnum; i++) {
      prefix[i] += prefix[i - 1];
      debug[i] = prefix[i];
    }
    *out_size = prefix[tnum];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  int idx = 0;
  for (int i = id * work_per_thread; i < (id + 1) * work_per_thread; i++) {
    int out_idx = prefix[id];
    if (lt_filter(src[i], filter_value)) {
      out[out_idx + idx] = src[i];
      idx++;
    }
  }
}

void kernel prefix_local_test(global const int *src, global int *dst, int sz) {
  int id = get_global_id(0);

  local int prefix[32];

  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int index = lidx + lidy * get_local_size(0);
  prefix[index] = src[index];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 5; ++i) {
    if (index >= (1 << i)) {
      prefix[index] += prefix[index - (1 << i)];
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  dst[0] = 0;
  dst[index + 1] = prefix[index];
}
