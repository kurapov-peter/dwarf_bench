#include "options.hpp"

std::istream &operator>>(std::istream &in, RunOptions::DeviceType &dt) {
  std::string type;
  in >> type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](char c) { return std::tolower(c); });
  if (type == "cpu")
    dt = RunOptions::DeviceType::CPU;
  else if (type == "gpu")
    dt = RunOptions::DeviceType::GPU;
  else if (type == "igpu")
    dt = RunOptions::DeviceType::iGPU;
  else
    dt = RunOptions::DeviceType::Default;

  return in;
}