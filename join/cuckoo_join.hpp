#pragma once
#include "join.hpp"
#include "common/common.hpp"

class CuckooJoin : public Dwarf {
public:
  CuckooJoin();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};


