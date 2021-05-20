#include "scan/scan.hpp"
#include <CL/cl.hpp>
#include <algorithm>
#include <boost/program_options.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <numeric>

namespace {
static int buffer_size;

std::vector<int> prefix_sum_scalar(const std::vector<int> &v) {
  std::vector<int> out(v.size() + 1, 0);
  for (int i = 0; i < v.size(); ++i) {
    out[i + 1] = out[i] + v[i];
  }
  out.resize(out.size() - 1);
  return out;
}
} // namespace

// todo: extend with a device type
template <int BUF_SIZE> class ScanTest : public ::testing::Test {
public:
  void SetUp() override {
    buffer_size = BUF_SIZE;
    // FIXME: path
    platform = oclhelpers::get_platform_matching("HD Graphics");
    std::tie(platform, device, ctx, program) =
        oclhelpers::compile_file_with_default_gpu(platform, "../scan/scan.cl");
    // std::tie(platform, device, ctx, program) =
    //     oclhelpers::compile_file_with_default_gpu("../scan/scan.cl");
  }

  void TearDown() override {}

protected:
  cl::Platform platform;
  cl::Device device;
  cl::Context ctx;
  cl::Program program;
};

TEST(ScanUtilsTest, PrefixSum) {
  std::vector<int> in = {0, 1, 1, 0, 0, 1, 1};
  std::vector<int> expected = {0, 0, 1, 2, 2, 2, 3};
  std::vector<int> std_out;

  ASSERT_EQ(expected, prefix_sum_scalar(in));
  /*std::inclusive_scan(in.begin(), in.end(),
                      std::inserter(std_out, std_out.begin()));
  ASSERT_EQ(expected, std_out);*/
}

using ScanTest32 = ScanTest<32>;

TEST_F(ScanTest32, PrefixSumLocal) {
  cl::CommandQueue queue(ctx, device);
  cl::Kernel kernel = cl::Kernel(program, "prefix_local_test");

  auto buffer_size_bytes = buffer_size * sizeof(int);
  cl::Buffer src(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer out(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);
  cl::Buffer prefix(ctx, CL_MEM_READ_WRITE, buffer_size_bytes);

  oclhelpers::set_args(kernel, src, out, /*prefix,*/ buffer_size);
  //   kernel.setArg(3, prefix);
  std::vector<int> host_src = helpers::make_random_uniform_binary(buffer_size);
  //   std::vector<int> host_src(buffer_size, 1);
  std::vector<int> host_out(buffer_size, -1);

  std::cout << "Memory supported: "
            << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << "Kb"
            << std::endl;

  OCL_SAFE_CALL(queue.enqueueWriteBuffer(src, CL_TRUE, 0, buffer_size_bytes,
                                         host_src.data()));
  OCL_SAFE_CALL(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                           cl::NDRange(buffer_size)));
  OCL_SAFE_CALL(queue.finish());
  OCL_SAFE_CALL(queue.enqueueReadBuffer(out, CL_TRUE, 0, buffer_size_bytes,
                                        host_out.data()));

  auto expected = prefix_sum_scalar(host_src);
  std::cout << "Input: ";
  for (auto &x : host_src) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  ASSERT_EQ(host_out, expected);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}