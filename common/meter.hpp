#pragma once
#include "result.hpp"

class Report {}; //?

class Meter {
public:
  Meter(const std::string &dwarf_name, MeasureResults &result)
      : dwarf_name_(dwarf_name), result_(result) {}
  void add_result(Result &&result);

private:
  const std::string dwarf_name_;
  MeasureResults &result_;
};