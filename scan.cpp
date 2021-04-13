#include "common.h"
#include <CL/cl.hpp>
#include <algorithm>
#include <vector>

namespace {
std::vector<int> expected_out_lt(std::vector<int> &v, int filter_value) {
  std::vector<int> out;
  std::copy_if(v.begin(), v.end(), std::back_inserter(out),
               [filter_value](const int v) { return v < filter_value; });
  return out;
}
} // namespace

int main() {
  auto platform = get_default_platform();
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;
  auto device = get_default_device(platform);
  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Context ctx({device});

  auto program = make_program_from_file(ctx, "scan.cl");
  if (program.build({device}) != CL_SUCCESS) {
    std::cerr << "Building failed: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    return 1;
  }

  const int buffer_size = 8;
  const int buffer_size_bytes = sizeof(int) * buffer_size;
  const int filter_value = 5;

  std::vector<int> host_out_size = {-1};

  cl::Buffer src(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer out(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer prefix(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer debug(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer out_size(ctx, CL_MEM_READ_WRITE, sizeof(int));

  cl::CommandQueue queue(ctx, device);

  std::vector<int> host_src = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> host_out(buffer_size, -1);
  std::vector<int> host_debug(buffer_size, -1);
  std::vector<int> expected_out = expected_out_lt(host_src, filter_value);

  queue.enqueueWriteBuffer(src, CL_TRUE, 0, buffer_size_bytes, host_src.data());

  cl::Kernel scan_kernel = cl::Kernel(program, "simple_two_pass_scan");
  scan_kernel.setArg(0, src);
  scan_kernel.setArg(1, buffer_size);
  scan_kernel.setArg(2, out);
  scan_kernel.setArg(3, out_size);
  scan_kernel.setArg(4, filter_value);
  scan_kernel.setArg(5, prefix);
  scan_kernel.setArg(6, debug);

  queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange,
                             cl::NDRange(buffer_size));

  queue.finish();
  queue.enqueueReadBuffer(out, CL_TRUE, 0, buffer_size_bytes, host_out.data());
  queue.enqueueReadBuffer(out_size, CL_TRUE, 0, sizeof(int),
                          host_out_size.data());
  queue.enqueueReadBuffer(debug, CL_TRUE, 0, buffer_size_bytes,
                          host_debug.data());

  std::cout << "Result size: ";
  dump_collection(host_out_size);
  host_out.resize(host_out_size[0]);
  std::cout << "Result:   ";
  dump_collection(host_out);
  std::cout << "Expected: ";
  dump_collection(expected_out);
  std::cout << "Debug: ";
  dump_collection(host_debug);
}