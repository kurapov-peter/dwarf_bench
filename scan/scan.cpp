#include "scan.hpp"
#include <CL/cl.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <oclhelpers.hpp>
#include <sstream>
#include <vector>

namespace {
std::vector<int> expected_out_lt(std::vector<int> &v, int filter_value) {
  std::vector<int> out;
  std::copy_if(v.begin(), v.end(), std::back_inserter(out),
               [filter_value](const int v) { return v < filter_value; });
  return out;
}
} // namespace

TwoPassScan::TwoPassScan() : Dwarf("TwoPassScan") {}

void TwoPassScan::run_two_pass_scan(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  cl::Platform platform;
  cl::Device device;
  cl::Context ctx;
  cl::Program program;

  using namespace oclhelpers;
  std::cout << "# of platforms: " << get_platforms().size() << std::endl;
  switch (opts.device_ty) {
  case RunOptions::CPU:
    std::tie(platform, device, ctx, program) =
        compile_file_with_default_cpu(kernel_path_);
    break;
  case RunOptions::GPU:
    std::tie(platform, device, ctx, program) =
        compile_file_with_default_gpu(kernel_path_);
    break;
  case RunOptions::iGPU:
    platform = get_platform_matching("HD Graphics");
    std::tie(platform, device, ctx, program) =
        compile_file_with_default_gpu(platform, kernel_path_);
    break;

  default:
    throw std::logic_error("Unsupported device type");
    break;
  }

  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl
            << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cout << "Device max WG size: "
            << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cout << "Device max compute units : "
            << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  std::cout << "Device max WI sizes: ";
  for (auto &x : device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  constexpr int GPU_MAX_THREADS = 256;
  constexpr int CPU_MAX_THREADS = 8;

  const int buffer_size = buf_size;
  const int threadnum =
      (opts.device_ty == RunOptions::CPU) ? CPU_MAX_THREADS : GPU_MAX_THREADS;
  const int buffer_size_bytes = sizeof(int) * buffer_size;
  const int prefix_size_bytes = sizeof(int) * (threadnum + 1);
  const int filter_value = 5;

  std::vector<int> host_out_size = {-1};

#ifdef CUDA_OCL_API_CHANGED
  const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
                                       CL_QUEUE_PROFILING_ENABLE, 0};
#else
  const auto props = CL_QUEUE_PROFILING_ENABLE;
#endif

  cl_int queue_init_err;
  cl::CommandQueue queue(ctx, device, props, &queue_init_err);
  if (queue_init_err != CL_SUCCESS) {
    std::cerr << "Queue init error: ";
    std::cerr << get_error_string(queue_init_err) << std::endl;
  }

  std::vector<int> host_src = helpers::make_random(buffer_size);

  for (auto it = 0; it < opts.iterations; ++it) {
    cl::Buffer src(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
    cl::Buffer out(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
    cl::Buffer prefix(ctx, CL_MEM_READ_WRITE, prefix_size_bytes);
    cl::Buffer debug(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
    cl::Buffer out_size(ctx, CL_MEM_READ_WRITE, sizeof(int));

    std::vector<int> host_out(buffer_size, -1);
    std::vector<int> host_debug(buffer_size, -1);

    cl::Kernel scan_kernel = cl::Kernel(program, "simple_two_pass_scan");
    oclhelpers::set_args(scan_kernel, src, buffer_size, out, out_size,
                         filter_value, prefix, debug);

    auto host_start = std::chrono::steady_clock::now();
    OCL_SAFE_CALL(queue.enqueueWriteBuffer(src, CL_TRUE, 0, buffer_size_bytes,
                                           host_src.data()));

    auto event = std::make_unique<cl::Event>();
    OCL_SAFE_CALL(queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange,
                                             cl::NDRange(threadnum),
                                             cl::NullRange, {}, event.get()));

    event->wait();
    OCL_SAFE_CALL(queue.enqueueReadBuffer(out, CL_TRUE, 0, buffer_size_bytes,
                                          host_out.data()));
    OCL_SAFE_CALL(queue.enqueueReadBuffer(out_size, CL_TRUE, 0, sizeof(int),
                                          host_out_size.data()));
#ifndef NDEBUG
    OCL_SAFE_CALL(queue.enqueueReadBuffer(debug, CL_TRUE, 0, buffer_size_bytes,
                                          host_debug.data()));
#endif
    OCL_SAFE_CALL(queue.finish());
    OCL_SAFE_CALL(queue.flush());

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    auto status = event->getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
      std::cout << "Status is " << status << " " << get_error_string(status)
                << " (should be CL_COMPLETE)";
    }

    Result result;
    result.host_time = host_end - host_start;

    cl_int profiling_error1;
    cl_int profiling_error2;
    auto exe_time =
        event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&profiling_error1) -
        event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&profiling_error2);
    if (profiling_error1 != CL_SUCCESS || profiling_error2 != CL_SUCCESS) {
      std::cerr << "Got profiling error: ";
      std::cerr << get_error_string(profiling_error1) << " & "
                << get_error_string(profiling_error2) << std::endl;
      result.valid = false;
    }

    result.kernel_time = exe_time;

    // todo: move out
    std::vector<int> expected_out = expected_out_lt(host_src, filter_value);
    size_t out_sz = host_out_size[0];
    host_out.resize(out_sz);

    if (expected_out != host_out) {
      std::cerr << "incorrect results" << std::endl;
      result.valid = false;
    }
    DwarfParams params{{"buf_size", std::to_string(buffer_size)}};
    meter.add_result(std::move(params), std::move(result));

#ifndef NDEBUG
    std::cout << "Input:    ";
    dump_collection(host_src);
    std::cout << "Result size: ";
    dump_collection(host_out_size);
    std::cout << "Result:   ";
    dump_collection(host_out);
    std::cout << "Expected result size: " << expected_out.size() << "\n";
    std::cout << "Expected: ";
    dump_collection(expected_out);
    std::cout << "Debug: ";
    dump_collection(host_debug);
#endif
  }
}

void TwoPassScan::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_two_pass_scan(size, meter());
  }
}

void TwoPassScan::init(const RunOptions &opts) {
  kernel_path_ = opts.root_path + "/scan/scan.cl";
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}