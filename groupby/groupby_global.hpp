#pragma once
#include "common/common.hpp"

class GroupByGlobal : public Dwarf {
public:
  GroupByGlobal();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};