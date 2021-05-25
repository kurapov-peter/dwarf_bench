#include "register_dwarfs.hpp"
#include "common/registry.hpp"
#include "constant/constant.hpp"
#include "radix/radix.hpp"
#include "scan/scan.hpp"

void populate_registry() {
  auto registry = Registry::instance();
  registry->registerd(new TwoPassScan());
  registry->registerd(new ConstantExample());
  registry->registerd(new ConstantExampleCAPI());

#ifdef DPCPP_ENABLED
  registry->registerd(new ConstantExampleDPCPP());
  registry->registerd(new ConstantExampleDPCPPCuda());
  registry->registerd(new DPLScan());
  registry->registerd(new DPLScanCuda());
  registry->registerd(new Radix());
  registry->registerd(new RadixCuda());
#endif
}