#include "constant.hpp"
#include <CL/cl.h>
#include <sstream>

ConstantExampleCAPI::ConstantExampleCAPI() : Dwarf("ConstantExampleCAPI") {}

namespace ocl = oclhelpers;
void ConstantExampleCAPI::run_constant(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  cl_platform_id plaftform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_kernel kernel;
  cl_program program;

  cl_int error;

  OCL_SAFE_CALL(clGetPlatformIDs(1, &plaftform, nullptr));
  OCL_SAFE_CALL(
      clGetDeviceIDs(plaftform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));
  context = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
  queue = clCreateCommandQueue(context, device, 0, &error);
  auto kernel_code = ocl::read_kernel_from_file(kernel_path_);
  const char *code = kernel_code.c_str();
  program = clCreateProgramWithSource(context, 1, (const char **)&code, nullptr,
                                      &error);

  OCL_SAFE_CALL(clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr));

  // Todo:
  //   std::cout << "Using platform: " << clGetPlatformInfo(plaftform,
  //   CL_PLATFORM_NAME, ) platform.getInfo<CL_PLATFORM_NAME>()
  //             << std::endl
  //             << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  const int buffer_size = buf_size;
  cl_mem src = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(int) * buffer_size, nullptr, &error);
  cl_mem dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(int) * buffer_size, nullptr, &error);
  std::vector<int> src_data(buffer_size, 11);
  std::vector<int> dst_data(buffer_size, -1);

  OCL_SAFE_CALL(clEnqueueWriteBuffer(queue, src, CL_TRUE, 0,
                                     sizeof(int) * buffer_size, src_data.data(),
                                     0, nullptr, nullptr));
  OCL_SAFE_CALL(clEnqueueWriteBuffer(queue, dst, CL_TRUE, 0,
                                     sizeof(int) * buffer_size, dst_data.data(),
                                     0, nullptr, nullptr));

  kernel = clCreateKernel(program, "constant_kernel", &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);

  size_t global_size = 1;
  size_t local_size = 1;
  cl_event event;
  clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_size, &local_size, 0,
                         nullptr, &event);
  clWaitForEvents(1, &event);

  clEnqueueReadBuffer(queue, src, CL_TRUE, 0, sizeof(int) * buffer_size,
                      src_data.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(int) * buffer_size,
                      dst_data.data(), 0, nullptr, nullptr);

  clFinish(queue);

  std::cout << dst_data[0] << " = "
            << "42\n";

  clReleaseMemObject(src);
  clReleaseMemObject(dst);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void ConstantExampleCAPI::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_constant(size, meter());
  }
}

void ConstantExampleCAPI::init(const RunOptions &opts) {
  kernel_path_ = opts.root_path + "/constant/constant.cl";
  meter().set_opts(opts);
}