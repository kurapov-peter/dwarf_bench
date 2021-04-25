#pragma once
#include <CL/cl.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <oclhelpers.hpp>

#include "dwarf.hpp"
#include "meter.hpp"
#include "options.hpp"

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

namespace helpers {
std::vector<int> make_random(size_t size);
std::string get_kernels_root_env(const char *argv0);
} // namespace helpers