#include <bench.hpp>
#include <iostream>

int main() {
  std::vector<DwarfBench::DeviceType> devices = {
    DwarfBench::DeviceType::CPU,
    DwarfBench::DeviceType::GPU,
  };

  std::vector<DwarfBench::Dwarf> dwarfs = {
    DwarfBench::Dwarf::Join,
    DwarfBench::Dwarf::Sort,
    DwarfBench::Dwarf::Scan,
    DwarfBench::Dwarf::GroupBy
  };
  
  DwarfBench::DwarfBench db;

  for (DwarfBench::Dwarf dwarf: dwarfs) {
    for (DwarfBench::DeviceType device: devices) {
      DwarfBench::RunConfig rc = {
          .device = device,
          .inputSize = 1024,
          .iterations = 10,
          .dwarf = dwarf,
      };

      auto results = db.makeMeasurements(rc);

      for (auto &result : results) {
        std::cout << dwarf << ' ' << device << " RESULT: " << result.dataSize << ' ' << result.microseconds
                  << std::endl;
      }
    }
  }

}
