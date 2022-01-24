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

  // For groupby
  size_t groups_count = 1;
  size_t executors = 1;
};

std::istream &operator>>(std::istream &in, RunOptions::DeviceType &dt);

std::string to_string(const RunOptions::DeviceType &dt);