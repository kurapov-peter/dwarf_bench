#include <bench.hpp>
#include <iostream>

int main() {
  DwarfBench::DwarfBench db;

  DwarfBench::RunConfig rc = {
      .device = DwarfBench::DeviceType::CPU,
      .inputSize = 1024,
      .iterations = 10,
      .dwarf = DwarfBench::Dwarf::TBBSort,
  };

  auto results = db.makeMeasurements(rc);

  for (auto &result : results) {
    std::cout << "RESULT: " << result.dataSize << ' ' << result.microseconds
              << std::endl;
  }
}
