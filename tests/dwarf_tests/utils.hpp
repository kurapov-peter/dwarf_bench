#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "common/options.hpp"

std::unique_ptr<RunOptions> get_cpu_test_opts();
std::unique_ptr<RunOptions> get_gpu_test_opts();

std::unique_ptr<RunOptions> get_cpu_test_opts_groupby();
std::unique_ptr<RunOptions> get_gpu_test_opts_groupby();

std::string get_kernels_root_tests();

bool not_cuda_gpu_available();

class StdoutCapture {
public:
  StdoutCapture();
  ~StdoutCapture();

private:
  std::stringstream buffer;
  std::streambuf *old_rdbuf;
};