#include "constant.hpp"
#include <CL/cl.hpp>
#include <oclhelpers.hpp>
#include <sstream>

ConstantExample::ConstantExample() : Dwarf("ConstantExample") {}

namespace ocl = oclhelpers;
void ConstantExample::run_constant(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  cl::Platform platform;
  cl::Device device;
  cl::Context ctx;
  cl::Program program;

  switch (opts.device_ty) {
  case RunOptions::CPU:
    std::tie(platform, device, ctx, program) =
        ocl::compile_file_with_default_cpu(kernel_path_);
    break;
  case RunOptions::GPU:
    std::tie(platform, device, ctx, program) =
        ocl::compile_file_with_default_gpu(kernel_path_);
    break;
  case RunOptions::iGPU:
    platform = ocl::get_platform_matching("HD Graphics");
    std::tie(platform, device, ctx, program) =
        ocl::compile_file_with_default_gpu(platform, kernel_path_);
    break;

  default:
    throw std::logic_error("Unsupported device type");
    break;
  }

  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl
            << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  const int buffer_size = buf_size;
  cl::Buffer src(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer dst(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  std::vector<int> src_data(buffer_size, 11);
  std::vector<int> dst_data(buffer_size, -1);

  cl::CommandQueue queue(ctx, device);
  OCL_SAFE_CALL(queue.enqueueWriteBuffer(
      src, CL_TRUE, 0, sizeof(int) * buffer_size, src_data.data()));
  OCL_SAFE_CALL(queue.enqueueWriteBuffer(
      dst, CL_TRUE, 0, sizeof(int) * buffer_size, dst_data.data()));
  cl::Kernel kernel = cl::Kernel(program, "constant_kernel");
  ocl::set_args(kernel, src, dst);

  cl::Event event;
  OCL_SAFE_CALL(queue.enqueueNDRangeKernel(
      kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, {}, &event));
  event.wait();
  OCL_SAFE_CALL(queue.enqueueReadBuffer(
      src, CL_TRUE, 0, sizeof(int) * buffer_size, src_data.data()));
  OCL_SAFE_CALL(queue.enqueueReadBuffer(
      dst, CL_TRUE, 0, sizeof(int) * buffer_size, dst_data.data()));
  OCL_SAFE_CALL(queue.finish());
  std::cout << dst_data[0] << " = "
            << "42\n";
}

void ConstantExample::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_constant(size, meter());
  }
}

void ConstantExample::init(const RunOptions &opts) {
  kernel_path_ = opts.root_path + "/constant/constant.cl";
  meter().set_opts(opts);
}