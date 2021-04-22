#pragma once
#include <chrono>
#include <ostream>
#include <string>
#include <vector>

using Duration = std::chrono::duration<double, std::micro>;
struct Result {
  size_t thread_x = 1, thread_y = 1, tread_z = 1;
  size_t group_size = 1;
  size_t bytes = 0;
  size_t iterations = 0;
  size_t bytes_per_iteration = 0;
  unsigned long kernel_time = 0;
  Duration host_time;
};

std::ostream &operator<<(std::ostream &os, const Result &res);

using SingleRunResults = std::vector<Result>;
class MeasureResults {
public:
  using const_iterator = SingleRunResults::const_iterator;
  MeasureResults(const std::string &name) : name_(name) {}

  void add_result(Result &&result);

  const_iterator begin() const;
  const_iterator end() const;

private:
  SingleRunResults results_;
  const std::string name_;
};
