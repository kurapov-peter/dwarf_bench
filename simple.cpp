#include "common/common.hpp"
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define LOG(what)                                                              \
  { std::cout << (what) << std::endl; }

namespace {

std::vector<int> populate_data(size_t sz) {
  std::vector<int> res(sz);
  for (size_t i = 0; i < sz; ++i) {
    res[i] = i;
  }
  return res;
}

std::vector<int> populate_poison(size_t sz) {
  std::vector<int> res;
  res.assign(sz, -1);
  return res;
}

} // namespace

int main() {
  using namespace oclhelpers;
  auto platform = get_default_platform();
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;
  auto device = get_default_device(platform);
  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Context ctx({device});

  auto program = make_program_from_file(ctx, "vadd.cl");
  if (program.build({device}) != CL_SUCCESS) {
    std::cerr << "Building failed: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    return 1;
  }

  constexpr int buffer_size = 10;

  cl::Buffer src1(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer src2(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer out(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);

  auto host_src1 = populate_data(buffer_size);
  auto host_src2 = populate_data(buffer_size);
  auto host_out = populate_poison(buffer_size);

  cl::CommandQueue queue(ctx, device);

  queue.enqueueWriteBuffer(src1, CL_TRUE, 0, sizeof(int) * buffer_size,
                           host_src1.data());
  queue.enqueueWriteBuffer(src2, CL_TRUE, 0, sizeof(int) * buffer_size,
                           host_src2.data());

  cl::Kernel vadd_kernel = cl::Kernel(program, "vadd");
  vadd_kernel.setArg(0, src1);
  vadd_kernel.setArg(1, src2);
  vadd_kernel.setArg(2, out);

  queue.enqueueNDRangeKernel(vadd_kernel, cl::NullRange, cl::NDRange(10),
                             cl::NullRange);

  queue.finish();
  queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(int) * buffer_size,
                          host_out.data());

  dump_collection(host_out);
}