#include "utils.hpp"

#include <boost/dll.hpp>

#include "common/dpcpp/dpcpp_common.hpp"
#include "common/result.hpp"

bool not_cuda_gpu_available() {
  sycl::device d;

  try {
    d = sycl::device(sycl::gpu_selector());
    return !is_cuda(d);
  } catch (sycl::exception const &e) {
    return false;
  }
}

std::unique_ptr<RunOptions> get_gpu_test_opts(size_t size) {
  RunOptions opts = {.device_ty = RunOptions::DeviceType::GPU,
                     .input_size = {size},
                     .iterations = 10,
                     .root_path = get_kernels_root_tests(),
                     .report_path = ""};

  return std::make_unique<RunOptions>(opts);
}

std::unique_ptr<RunOptions> get_cpu_test_opts(size_t size) {
  RunOptions opts = {.device_ty = RunOptions::DeviceType::CPU,
                     .input_size = {size},
                     .iterations = 10,
                     .root_path = get_kernels_root_tests(),
                     .report_path = ""};

  return std::make_unique<RunOptions>(opts);
}

std::unique_ptr<RunOptions> get_cpu_test_opts_groupby(size_t size) {
  return std::make_unique<GroupByRunOptions>(*get_cpu_test_opts(size), 64,
                                             1024);
}

std::unique_ptr<RunOptions> get_gpu_test_opts_groupby(size_t size) {
  return std::make_unique<GroupByRunOptions>(*get_gpu_test_opts(size), 64,
                                             1024);
}

std::string get_kernels_root_tests() {
  auto *val = std::getenv("DWARF_BENCH_ROOT");
  return val ? val
             : boost::dll::program_location()
                   .parent_path()
                   .parent_path()
                   .parent_path()
                   .c_str();
}

StdoutCapture::StdoutCapture() {
  old_rdbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
}

StdoutCapture::~StdoutCapture() { std::cout.rdbuf(old_rdbuf); }