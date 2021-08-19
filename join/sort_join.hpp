#pragma once
#include "join.hpp"
#include "common/common.hpp"

class SortJoin : public Dwarf {
public:
  SortJoin();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};


