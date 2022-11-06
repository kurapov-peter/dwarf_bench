#pragma once

#include "common/registry.hpp"
#include "register_dwarfs.hpp"

#include "common/common.hpp"

#include <vector>

struct Measurement {
  size_t dataSize;
  size_t microseconds;
};

struct RunConfig {
  std::unique_ptr<RunOptions> opts;
  std::string dwarf;
};

class DwarfBench {
public:
  DwarfBench();

  std::vector<Measurement> makeMeasurements(const RunConfig &conf);

private:
  Registry *reg;
};
