#include "constant.hpp"
#include <CL/cl.hpp>
#include <sstream>

ConstantExample::ConstantExample() : Dwarf("ConstantExample") {}

namespace ocl = oclhelpers;
void ConstantExample::run_constant(const size_t buffer_size, Meter &meter) {
  auto opts = meter.opts();
  auto [platform, device, ctx, program] =
      (opts.device_ty == RunOptions::CPU)
          ? ocl::compile_file_with_default_cpu(kernel_path_)
          : ocl::compile_file_with_default_gpu(kernel_path_);
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl
            << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Buffer src(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer dst(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  std::vector<int> src_data = {11};
  std::vector<int> dst_data = {-1};

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