#include "register_dwarfs.hpp"
#include "common/registry.hpp"
#include "constant/constant.hpp"
#include "scan/scan.hpp"

void populate_registry() {
  auto registry = Registry::instance();
  registry->registerd(new TwoPassScan());
  registry->registerd(new ConstantExample());
}