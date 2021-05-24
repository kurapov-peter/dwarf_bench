#include "common.hpp"
#include "boost/dll.hpp"
#include <cstdlib>

namespace helpers {
std::vector<int> make_random(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> out(size);
  std::uniform_int_distribution<int> dist(0, 10);
  std::generate(out.begin(), out.end(), [&]() { return dist(gen); });
  return out;
}

std::vector<int> make_random_uniform_binary(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> out(size);
  std::uniform_int_distribution<int> dist(0, 1);
  std::generate(out.begin(), out.end(), [&]() { return dist(gen); });
  return out;
}

std::string get_kernels_root_env(const char *argv0) {
  auto *val = std::getenv("DWARF_BENCH_ROOT");
  return val ? val : boost::dll::program_location().parent_path().c_str();
}

void set_dpcpp_filter_env(const RunOptions &opts) {
  switch (opts.device_ty) {
  case RunOptions::DeviceType::GPU:
    set_dpcpp_filter_env_no_overwrite("cuda");
    break;

  default:
    break;
  }
}

void set_dpcpp_filter_env_no_overwrite(const char *filter) {
  std::cout << "SETTING DEVICE FILTER to " << filter << "\n";
  setenv("SYCL_DEVICE_FILTER", filter, 1);
}
} // namespace helpers