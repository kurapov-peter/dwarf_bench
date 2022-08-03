#pragma once

#include <iostream>
#include <string>
#include <sstream>

#include "common/options.hpp"

std::string get_kernels_root_tests();
RunOptions get_cpu_test_opts();

RunOptions get_gpu_test_opts();
bool not_cuda_gpu_available();

class StdoutCapture {
public:
    StdoutCapture();
    ~StdoutCapture();

private:
    std::stringstream buffer;
    std::streambuf *old_rdbuf;
};