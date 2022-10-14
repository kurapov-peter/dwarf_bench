#pragma once
#include "common/common.hpp"
#include "groupby.hpp"

class GroupByGlobal : public GroupBy {
public:
  GroupByGlobal();

private:
  void _run(const size_t buffer_size, Meter &meter) override;
};
