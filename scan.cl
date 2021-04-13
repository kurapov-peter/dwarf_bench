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
  if (id != tnum - 1) {
    prefix[id + 1] = sz;
  }

  // prefix sum todo
  if (id == 0) {
    prefix[0] = 0;
    debug[0] = prefix[0];
    for (int i = 1; i < tnum; i++) {
      prefix[i] += prefix[i - 1];
      debug[i] = prefix[i];
    }
  }

  if (id == tnum - 1) {
    *out_size = prefix[id];
  }

  int idx = 0;
  for (int i = id * work_per_thread; i < (id + 1) * work_per_thread;
       i++, idx++) {
    int out_idx = prefix[id];
    if (lt_filter(src[i], filter_value)) {
      out[out_idx + idx] = src[i];
    }
  }
}
