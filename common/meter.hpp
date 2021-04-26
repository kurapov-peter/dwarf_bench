#pragma once
#include "options.hpp"
#include "result.hpp"

class Report {}; //?

class Meter {
public:
  Meter(const std::string &dwarf_name, MeasureResults &result)
      : dwarf_name_(dwarf_name), result_(result) {}
  void add_result(Result &&result);
  void set_params(DwarfParams params);
  void set_opts(const RunOptions &opts);
  RunOptions opts() const;

private:
  const std::string dwarf_name_;
  MeasureResults &result_;
  DwarfParams params_;
  RunOptions opts_;
};