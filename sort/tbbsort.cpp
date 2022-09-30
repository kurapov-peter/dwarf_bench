#include <oneapi/tbb/parallel_sort.h>

#include "sort/tbbsort.hpp"

namespace {
template <typename T> std::vector<T> expected_out(const std::vector<T> &v) {
  std::vector<int> out = v;
  std::sort(out.begin(), out.end());
  return out;
}
} // namespace

TBBSort::TBBSort() : Dwarf("TBBSort") {}

void TBBSort::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  std::vector<int> host_src = helpers::make_random<int>(buf_size);
  const std::vector<int> expected = expected_out(host_src);

  for (auto it = 0; it < opts.iterations; ++it) {
    auto host_start = std::chrono::steady_clock::now();
    oneapi::tbb::parallel_sort(host_src.begin(), host_src.end());
    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
#ifndef NDEBUG
    {
      std::cout << "Output:    ";
      dump_collection(host_src);
      std::cout << std::endl;
      std::cout << "Expected:  ";
      dump_collection(expected);
    }
#endif
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;
    DwarfParams params{{"buf_size", std::to_string(buf_size)}};

    {
      if (!helpers::check_first(host_src, expected, expected.size())) {
        std::cerr << "incorrect results" << std::endl;
        result->valid = false;
      }
    }
    meter.add_result(std::move(params), std::move(result));
  }
}

void TBBSort::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}

void TBBSort::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
