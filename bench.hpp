#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace DwarfBench {

/**
 * @brief Dwarfs list enum
 *
 */
enum Dwarf {
  ConstantExampleDPCPP,
  DPLScan,
  GroupBy,
  GroupByLocal,
  HashBuild,
  HashBuildNonBitmask,
  Join,
  NestedLoopJoin,
  Radix,
  TBBSort,
};

/**
 * @brief Device type for execution settings
 *
 */
enum DeviceType { CPU, GPU };

/**
 * @brief Measurements got from execution
 *
 * @var Measurement::dataSize how many bytes were used during execution
 * @var Measurement::microseconds how many microseconds did the execution last
 */
struct Measurement {
  size_t dataSize;
  size_t microseconds;
};

/**
 * @brief Execution configuration
 *
 * @var RunConfig::device on which device to be executed
 * @var RunConfig::inputSize data array size, ususally a column size in elements
 * @var RunConfig::iterations number of iterations to run a bmark
 * @var RunConfig::dwarf dwarf to run
 */
struct RunConfig {
  DeviceType device;
  size_t inputSize;
  size_t iterations;
  Dwarf dwarf;
};

/**
 * @brief main class for execution
 *
 */
class DwarfBench {
public:
  DwarfBench() = default;

  /**
   * @brief make measurements of a Dwarf based on configuration
   *
   * @param conf sets the generated data size, the number of iterations, the
   * device and the Dwarf to run
   * @return std::vector<Measurement> measurements made by dwarfs. Each element
   * corresponds to a single run.
   */
  std::vector<Measurement> makeMeasurements(const RunConfig &conf);
};

class DwarfBenchException : public std::exception {
private:
  std::string message_;

public:
  explicit DwarfBenchException(const std::string &message);
  const char *what() const noexcept override;
};

} // namespace DwarfBench
