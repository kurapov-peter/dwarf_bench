#include <gtest/gtest.h>
#include <memory>

#include "common/dpcpp/dpcpp_common.hpp"
#include "common/options.hpp"
#include "common/registry.hpp"
#include "common/result.hpp"

#include "dwarfs.hpp"
#include "utils.hpp"

void check_results(std::unique_ptr<Dwarf> dwarf) {
  for (const DwarfRunResult &res : dwarf->get_results()) {
    ASSERT_TRUE(res.result->valid);
  }
}

template <class DwarfClass> void test_dwarf(std::unique_ptr<RunOptions> opts) {
  std::unique_ptr<Dwarf> dwarf = std::make_unique<DwarfClass>();
  dwarf->init(*opts);
  dwarf->run(*opts);
  check_results(std::move(dwarf));
}

#define GENERATE_TEST_BASE(DWARF, CPU_OPTS, GPU_OPTS, SIZE)                    \
  TEST(DWARF##ExpectedCheck, SIZE_##SIZE##_CPU) {                              \
    StdoutCapture c;                                                           \
    test_dwarf<DWARF>(std::move(CPU_OPTS(SIZE)));                              \
  }                                                                            \
                                                                               \
  TEST(DWARF##ExpectedCheck, SIZE_##SIZE##_GPU) {                              \
    if (!not_cuda_gpu_available())                                             \
      GTEST_SKIP() << "No GPU without CUDA available";                         \
    StdoutCapture c;                                                           \
    test_dwarf<DWARF>(std::move(GPU_OPTS(SIZE)));                              \
  }

#define GENERATE_TEST(DWARF, SIZE)                                             \
  GENERATE_TEST_BASE(DWARF, get_cpu_test_opts, get_gpu_test_opts, SIZE)
#define GENERATE_GROUPBY_TEST(DWARF, SIZE)                                     \
  GENERATE_TEST_BASE(DWARF, get_cpu_test_opts_groupby,                         \
                     get_gpu_test_opts_groupby, SIZE)

#define GENERATE_TEST_SUITE(DWARF)                                             \
  GENERATE_TEST(DWARF, 128)                                                    \
  GENERATE_TEST(DWARF, 256)                                                    \
  GENERATE_TEST(DWARF, 512)                                                    \
  GENERATE_TEST(DWARF, 1024)                                                   \
  GENERATE_TEST(DWARF, 2048)                                                   \
  GENERATE_TEST(DWARF, 4096)

#define GENERATE_TEST_SUITE_GROUPBY(DWARF)                                     \
  GENERATE_GROUPBY_TEST(DWARF, 128)                                            \
  GENERATE_GROUPBY_TEST(DWARF, 256)                                            \
  GENERATE_GROUPBY_TEST(DWARF, 512)                                            \
  GENERATE_GROUPBY_TEST(DWARF, 1024)                                           \
  GENERATE_GROUPBY_TEST(DWARF, 2048)                                           \
  GENERATE_GROUPBY_TEST(DWARF, 4096)

GENERATE_TEST_SUITE(TwoPassScan);
GENERATE_TEST_SUITE(ConstantExample);
GENERATE_TEST_SUITE(ConstantExampleCAPI);
GENERATE_TEST_SUITE(TBBSort);

#ifdef DPCPP_ENABLED
GENERATE_TEST_SUITE(ConstantExampleDPCPP);
GENERATE_TEST_SUITE(DPLScan);
GENERATE_TEST_SUITE(Radix);
GENERATE_TEST_SUITE(ReduceDPCPP);
GENERATE_TEST_SUITE(HashBuild);
GENERATE_TEST_SUITE(NestedLoopJoin);
GENERATE_TEST_SUITE(CuckooHashBuild);
GENERATE_TEST_SUITE_GROUPBY(GroupBy);
GENERATE_TEST_SUITE_GROUPBY(GroupByLocal);
GENERATE_TEST_SUITE(Join);
GENERATE_TEST_SUITE(HashBuildNonBitmask);
GENERATE_TEST_SUITE(JoinOmnisci);
#ifdef EXPERIMENTAL
GENERATE_TEST_SUITE(SlabHashBuild);
GENERATE_TEST_SUITE(SlabJoin);
GENERATE_TEST_SUITE(SlabProbe);
#endif
#ifdef CUDA_ENABLED
GENERATE_TEST_SUITE(ConstantExampleDPCPPCuda);
GENERATE_TEST_SUITE(DPLScanCuda);
GENERATE_TEST_SUITE(RadixCuda);
#endif
#endif

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
