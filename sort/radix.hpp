#pragma once
#include "../common/common.hpp"

class Radix : public Dwarf {
public:
  Radix();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};

class RadixCuda : public Dwarf {
public:
  RadixCuda();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};