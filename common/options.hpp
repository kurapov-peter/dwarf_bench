#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

struct RunOptions {
  enum DeviceType { CPU, GPU, iGPU, Default };
  DeviceType device_ty = DeviceType::Default;
  std::vector<size_t> input_size;
  size_t iterations = 1;
  std::string root_path;
  std::string report_path;
};

struct GroupByRunOptions : public RunOptions {
  GroupByRunOptions(const RunOptions &opts, size_t groups_count,
                    size_t threads_count, size_t work_group_size)
      : RunOptions(opts), groups_count(groups_count), threads_count(threads_count), work_group_size(work_group_size) {};
  size_t groups_count;
  size_t threads_count;
  size_t work_group_size;
};

std::istream &operator>>(std::istream &in, RunOptions::DeviceType &dt);

std::string to_string(const RunOptions::DeviceType &dt);