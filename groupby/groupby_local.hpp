#pragma once
#include "common/common.hpp"

class GroupByLocal : public Dwarf {
public:
  GroupByLocal();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};
