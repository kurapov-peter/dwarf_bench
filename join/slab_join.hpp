#pragma once
#include "common/common.hpp"

class SlabJoin : public Dwarf {
public:
  SlabJoin();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};