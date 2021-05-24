#include "common/dpcpp/dpcpp_common.hpp"
#include "constant.hpp"
#include <CL/sycl.hpp>
#include <sstream>

ConstantExampleDPCPPCuda::ConstantExampleDPCPPCuda()
    : Dwarf("ConstantExampleDPCPPCuda") {}

namespace {
void _run_constant(const size_t buf_size, Meter &meter) {
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
} // namespace

void ConstantExampleDPCPPCuda::run_constant(const size_t buf_size,
                                            Meter &meter) {
  _run_constant(buf_size, meter);
}

void ConstantExampleDPCPPCuda::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_constant(size, meter());
  }
}

void ConstantExampleDPCPPCuda::init(const RunOptions &opts) {
  meter().set_opts(opts);
}