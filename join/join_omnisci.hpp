#pragma once
#include "common/common.hpp"

class JoinOmnisci : public Dwarf {
public:
  JoinOmnisci();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};

class JoinOmnisciCuda : public Dwarf {
public:
  JoinOmnisciCuda();
  void run(const RunOptions &opts) override;
  void init(const RunOptions &opts) override;

private:
  void _run(const size_t buffer_size, Meter &meter);
};
