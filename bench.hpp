#pragma once

#include <vector>
#include <string>
#include <stdexcept>

namespace DwarfBench {

/**
 * @brief Dwarfs list enum
 * 
 */
enum Dwarf {
  CONSTANT_EXAMPLE,
  CONSTANT_EXAMPLE_C_API,
  CONSTANT_EXAMPLE_DPCPP,
  CUCKOO_HASH_BUILD,
  DPL_SCAN,
  GROUP_BY,
  GROUP_BY_LOCAL,
  HASH_BUILD,
  HASH_BUILD_NON_BITMASK,
  JOIN,
  NESTED_LOOP_JOIN,
  RADIX,
  REDUCE_DPCPP,
  TBB_SORT,
  TWO_PASS_SCAN,
};

/**
 * @brief Device type for execution settings
 * 
 */
enum DeviceType {
  CPU,
  GPU
};

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
   * @brief make measurements based on configuration
   * 
   * @param conf execution configuration
   * @return std::vector<Measurement> measurements
   */
  std::vector<Measurement> makeMeasurements(const RunConfig &conf);
};

class DwarfBenchException: public std::exception {
private:
    std::string message_;
public:
    explicit DwarfBenchException(const std::string& message);
    const char* what() const noexcept override;
};

}