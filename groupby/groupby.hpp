#pragma once
#include "common/common.hpp"
#include <functional>

class GroupBy : public Dwarf {
public:
  GroupBy(const std::string &suffix);
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

protected:
  using AggregationFunc = std::function<uint32_t(uint32_t, uint32_t)>;
  AggregationFunc add = [](uint32_t acc, uint32_t x) { return acc + x; };
  AggregationFunc mul = [](uint32_t acc, uint32_t x) { return acc * x; };
  AggregationFunc count = [](uint32_t acc, uint32_t) { return acc + 1; };
  
  const uint32_t _empty_element = std::numeric_limits<uint32_t>::max();


  virtual void _run(const size_t buffer_size, Meter &meter) = 0;

  std::vector<uint32_t> src_keys;
  std::vector<uint32_t> src_vals;
  std::vector<uint32_t> expected;

  void generate_keys(size_t buf_size, size_t groups_count);
  void generate_vals(size_t buf_size);
  void generate_expected(size_t groups_count, AggregationFunc f);

  bool check_correctness(const std::vector<uint32_t> &result);
};