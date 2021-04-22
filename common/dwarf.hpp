#pragma once
#include "meter.hpp"
#include "options.hpp"
#include "result.hpp"

class Dwarf {
public:
  Dwarf(const std::string &name)
      : name_(name), results_(name), meter_(name, results_) {}
  virtual ~Dwarf() = default;

  const std::string &name() const { return name_; }

  virtual void run(const RunOptions &opts) = 0;
  virtual void init(const RunOptions &opts) = 0;
  void report(const RunOptions &opts) {
    for (const auto &res : results_) {
      // std::cout << res;
      opts.os << res;
    }
  }

  Meter &meter() { return meter_; }

private:
  std::string name_;
  MeasureResults results_;
  Meter meter_;
};