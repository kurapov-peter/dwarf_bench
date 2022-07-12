#include "groupby.hpp"
#include "common/dpcpp/hashtable.hpp"
#include <limits>

GroupBy::GroupBy(const std::string &suffix) : Dwarf("GroupBy" + suffix) {}

void GroupBy::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}

void GroupBy::init(const RunOptions &opts) {
  reporting_header_ = "total_time,group_by_time,reduction_time";
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}

void GroupBy::generate_expected(size_t groups_count, AggregationFunc f) {
  expected.resize(groups_count);
  size_t data_size = src_keys.size();

  for (int i = 0; i < data_size; i++) {
    expected[src_keys[i]] = f(expected[src_keys[i]], src_vals[i]);
  }
}

bool GroupBy::check_correctness(const std::vector<uint32_t> &result) {
  if (result != expected) {
    std::cerr << "Incorrect results" << std::endl;
    return false;
  }
  return true;
}

void GroupBy::generate_keys(size_t buf_size, size_t groups_count) {
  src_keys = helpers::make_random<uint32_t>(buf_size, 0, groups_count - 1);
}

void GroupBy::generate_vals(size_t buf_size) {
  src_vals = helpers::make_random<uint32_t>(buf_size);
}

size_t GroupBy::get_size(size_t buf_size) {
  return buf_size * 2;
}
