#include "meter.hpp"

void Meter::add_result(Result &&result) {
  result_.add_result(params_, std::move(result));
}

void Meter::set_params(DwarfParams params) { params_ = params; }

void Meter::set_opts(const RunOptions &opts) { opts_ = opts; }
RunOptions Meter::opts() const { return opts_; }
