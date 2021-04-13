#pragma once
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

cl::Platform get_default_platform() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (!platforms.size()) {
    std::cerr << "No platform found, exiting...\n";
    exit(1);
  }
  return platforms[0];
}

std::vector<cl::Device> get_gpus(const cl::Platform &p) {
  std::vector<cl::Device> devices;
  p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (!devices.size()) {
    std::cerr << "No gpus found, exiting...\n";
    exit(1);
  }
  return devices;
}

cl::Device get_default_device(const cl::Platform &p) { return get_gpus(p)[0]; }

std::string read_kernel_from_file(const std::string &filename) {
  std::ifstream is(filename);
  return std::string(std::istreambuf_iterator<char>(is),
                     std::istreambuf_iterator<char>());
}

cl::Program make_program_from_file(cl::Context &ctx,
                                   const std::string &filename) {
  cl::Program::Sources sources;
  auto kernel_code = read_kernel_from_file(filename);
  return {ctx, kernel_code};
}

template <class Collection>
void dump_collection(const Collection &c, std::ostream &os = std::cout) {
  bool first = true;
  for (const auto &e : c) {
    if (!first) {
      os << " ";
    }
    os << e;
    first = false;
  }
  os << "\n";
}