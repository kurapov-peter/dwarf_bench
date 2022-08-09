#include "dpcpp_common.hpp"

std::unique_ptr<cl::sycl::device_selector>
get_device_selector(const RunOptions &opts) {
  using namespace cl::sycl;
  switch (opts.device_ty) {
  case RunOptions::DeviceType::CPU:
    return std::make_unique<cpu_selector>();
  case RunOptions::DeviceType::GPU:
  case RunOptions::DeviceType::iGPU:
    return std::make_unique<gpu_selector>();

  case RunOptions::DeviceType::Default:
    return std::make_unique<default_selector>();

  default:
    throw std::logic_error("Unsupported device type.");
  }
}

bool is_cuda(const sycl::device &d) {
  const static pi_uint32 nvidia_vendor_id = 0x10DE;
  return d.get_info<sycl::info::device::vendor_id>() == nvidia_vendor_id;
}
