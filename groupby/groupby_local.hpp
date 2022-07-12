#pragma once
#include "common/common.hpp"
#include "groupby.hpp"

class GroupByLocal : public GroupBy {
public:
  GroupByLocal();

private:
  void _run(const size_t buffer_size, Meter &meter);
};
