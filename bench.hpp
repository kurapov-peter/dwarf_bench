#pragma once

#include <vector>
#include <string>

namespace DwarfBench {

enum DeviceType {
  CPU,
  GPU
};

struct Measurement {
  size_t dataSize;
  size_t microseconds;
};

struct RunConfig {
  DeviceType device;
  size_t inputSize;
  size_t iterations;
  std::string dwarf;
};

class DwarfBench {
public:
  DwarfBench() = default;

  std::vector<Measurement> makeMeasurements(const RunConfig &conf);
};

}