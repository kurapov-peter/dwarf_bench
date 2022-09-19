#pragma once
#include "../common/common.hpp"

class ConstantExample : public Dwarf {
public:
  ConstantExample();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void run_constant(const size_t buffer_size, Meter &meter);
  std::string kernel_path_;
};

class ConstantExampleCAPI : public Dwarf {
public:
  ConstantExampleCAPI();
  void run(const RunOptions &) override;
  void init(const RunOptions &) override;

private:
  void run_constant(const size_t buffer_size, Meter &meter);
  std::string kernel_path_;
};

class ConstantExampleDPCPP : public Dwarf {
public:
  ConstantExampleDPCPP();
  void run(const RunOptions &) override;
  void init(const RunOptions &) override;

private:
  void run_constant(const size_t buffer_size, Meter &meter);
  std::string kernel_path_;
};

class ConstantExampleDPCPPCuda : public Dwarf {
public:
  ConstantExampleDPCPPCuda();
  void run(const RunOptions &) override;
  void init(const RunOptions &) override;

private:
  void run_constant(const size_t buffer_size, Meter &meter);
  std::string kernel_path_;
};