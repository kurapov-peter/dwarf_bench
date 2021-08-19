#pragma once
#include <chrono>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <memory>

using DwarfParams = std::map<std::string, std::string>;

using Duration = std::chrono::duration<double, std::micro>;
struct Result {
  size_t thread_x = 1, thread_y = 1, tread_z = 1;
  size_t group_size = 1;
  size_t bytes = 0;
  size_t iterations = 0;
  size_t bytes_per_iteration = 0;
  unsigned long kernel_time = 0;
  Duration host_time;
  bool valid = true;

protected:
  virtual std::ostream &format(std::ostream &os) const;
  friend std::ostream& operator << (std::ostream& out, const Result& instance);
};

struct HashJoinResult : public Result {
  Duration probe_time;
  Duration build_time;
  std::ostream &format(std::ostream &os) const override;
};

std::ostream &operator<<(std::ostream &os, const Result &res);

struct DwarfRunResult {
  DwarfParams params;
  std::unique_ptr<Result> result;
};

using SingleRunResults = std::vector<DwarfRunResult>;
class MeasureResults {
public:
  using const_iterator = SingleRunResults::const_iterator;
  MeasureResults(const std::string &name) : name_(name) {}

  void add_result(DwarfParams params, std::unique_ptr<Result> result);

  const_iterator begin() const;
  const_iterator end() const;

  void write_csv(const std::string &filename) const;

private:
  SingleRunResults results_;
  const std::string name_;
};
