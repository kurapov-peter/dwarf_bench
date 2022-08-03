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

#define GENERATE_TEST_BASE(DWARF, CPU_OPTS, GPU_OPTS)                          \
  TEST(DwarfsExpectedCorrectnessTests, DWARF##_CPU) {                          \
    StdoutCapture c;                                                           \
    test_dwarf<DWARF>(std::move(CPU_OPTS()));                                  \
  }                                                                            \
                                                                               \
  TEST(DwarfsExpectedCorrectnessTests, DWARF##_GPU) {                          \
    if (!not_cuda_gpu_available())                                             \
      GTEST_SKIP() << "No GPU without CUDA available";                         \
    StdoutCapture c;                                                           \
    test_dwarf<DWARF>(std::move(GPU_OPTS()));                                  \
  }

#define GENERATE_TEST(DWARF)                                                   \
  GENERATE_TEST_BASE(DWARF, get_cpu_test_opts, get_gpu_test_opts)
#define GENERATE_GROUPBY_TEST(DWARF)                                           \
  GENERATE_TEST_BASE(DWARF, get_cpu_test_opts_groupby,                         \
                     get_gpu_test_opts_groupby)

GENERATE_TEST(TwoPassScan);
GENERATE_TEST(ConstantExample);
GENERATE_TEST(ConstantExampleCAPI);
GENERATE_TEST(TBBSort);

#ifdef DPCPP_ENABLED
GENERATE_TEST(ConstantExampleDPCPP);
GENERATE_TEST(DPLScan);
GENERATE_TEST(Radix);
GENERATE_TEST(ReduceDPCPP);
GENERATE_TEST(HashBuild);
GENERATE_TEST(NestedLoopJoin);
GENERATE_TEST(CuckooHashBuild);
GENERATE_GROUPBY_TEST(GroupBy);
GENERATE_GROUPBY_TEST(GroupByLocal);
GENERATE_TEST(Join);
GENERATE_TEST(HashBuildNonBitmask);
GENERATE_TEST(JoinOmnisci);
#ifdef EXPERIMENTAL
GENERATE_TEST(SlabHashBuild);
GENERATE_TEST(SlabJoin);
GENERATE_TEST(SlabProbe);
#endif
#ifdef CUDA_ENABLED
GENERATE_TEST(ConstantExampleDPCPPCuda);
GENERATE_TEST(DPLScanCuda);
GENERATE_TEST(RadixCuda);
#endif
#endif

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}