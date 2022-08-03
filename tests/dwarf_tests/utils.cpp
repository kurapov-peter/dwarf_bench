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

RunOptions get_gpu_test_opts() {
    RunOptions opts = {
        .device_ty = RunOptions::DeviceType::GPU,
        .input_size = { 128, 256, 512, 1024, 2048, 4096 },
        .iterations = 10,
        .root_path = get_kernels_root_tests(),
        .report_path = ""
    };

    return opts;
}

RunOptions get_cpu_test_opts() {
    RunOptions opts = {
        .device_ty = RunOptions::DeviceType::CPU,
        .input_size = { 128, 256, 512, 1024, 2048, 4096 },
        .iterations = 10,
        .root_path = get_kernels_root_tests(),
        .report_path = ""
    };

    return opts;
}

std::string get_kernels_root_tests() {
  auto *val = std::getenv("DWARF_BENCH_ROOT");
  return val ? val : boost::dll::program_location().parent_path().parent_path().parent_path().c_str();
}

StdoutCapture::StdoutCapture() {
    old_rdbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
}

StdoutCapture::~StdoutCapture() {
    std::cout.rdbuf(old_rdbuf);
}