#include "constant.hpp"
#include <CL/sycl.hpp>
#include <sstream>

namespace {
using namespace cl::sycl;
std::unique_ptr<device_selector> get_device_selector(const RunOptions &opts) {
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
} // namespace

ConstantExampleDPCPP::ConstantExampleDPCPP() : Dwarf("ConstantExampleDPCPP") {}

void ConstantExampleDPCPP::run_constant(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  using namespace cl::sycl;

  auto sel = get_device_selector(opts);

  constexpr int num = 16;
  auto rng = range<1>{num};

  buffer<int> src{rng};

  queue q{*sel.get()};
  std::cout << "Selected device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  q.submit([&](handler &h) {
    auto out = src.template get_access<access::mode::write>(h);
    h.parallel_for<class hello>(rng, [=](auto &idx) { out[idx] = 42; });
  });

  auto result = src.get_access<access::mode::read>();

  std::cout << result[0] << " = "
            << "42\n";
}

void ConstantExampleDPCPP::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_constant(size, meter());
  }
}

void ConstantExampleDPCPP::init(const RunOptions &opts) {
  meter().set_opts(opts);
}