#pragma once
#include <iostream>
#include <vector>

struct RunOptions {
  enum DeviceType { CPU, GPU, Default };
  DeviceType device_ty = DeviceType::Default;
  std::vector<size_t> input_size;
  size_t iterations = 1;
  std::string root_path;
};