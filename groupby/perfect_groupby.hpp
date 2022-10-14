#pragma once
#include "common/common.hpp"
#include "groupby.hpp"

class PerfectGroupBy : public GroupBy {
public:
  PerfectGroupBy();

private:
  void _run(const size_t buffer_size, Meter &meter) override;
  size_t get_size(size_t buf_size) override;
};
