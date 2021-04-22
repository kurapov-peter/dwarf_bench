#pragma once
#include "common/common.hpp"

class TwoPassScan : public Dwarf {
public:
  TwoPassScan();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void run_two_pass_scan(const size_t buffer_size, Meter &meter);
  std::string kernel_path_;
};
