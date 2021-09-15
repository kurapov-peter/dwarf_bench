#include "meter.hpp"

DwarfParams concat(const DwarfParams &stable, DwarfParams &&incoming) {
  DwarfParams out = stable;
  out.insert(incoming.begin(), incoming.end());
  return out;
}

void Meter::add_result(DwarfParams &&params, std::unique_ptr<Result> result) {
  result_.add_result(concat(params_, std::move(params)), std::move(result));
}

void Meter::set_params(DwarfParams params) { params_ = params; }

void Meter::set_opts(const RunOptions &opts) { opts_ = opts; }
const RunOptions &Meter::opts() const { return opts_; }
