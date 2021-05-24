#include "common/options.hpp"
#include <CL/sycl.hpp>

std::unique_ptr<cl::sycl::device_selector>
get_device_selector(const RunOptions &opts);