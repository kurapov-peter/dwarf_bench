#pragma once
#include <chrono>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

using DwarfParams = std::map<std::string, std::string>;

using Duration = std::chrono::duration<double, std::micro>;
struct Result {
  size_t thread_x = 1, thread_y = 1, tread_z = 1;
  size_t group_size = 1;
  size_t bytes = 0;
  size_t iterations = 0;
  size_t bytes_per_iteration = 0;
  Duration kernel_time;
  Duration host_time;
  bool valid = true;

  virtual std::vector<Duration> get_reported_timings_list() const;

protected:
  virtual std::ostream &print_to_stream(std::ostream &os) const;
  friend std::ostream &operator<<(std::ostream &out, const Result &instance);
};

struct HashJoinResult : public Result {
  Duration probe_time;
  Duration build_time;
  std::ostream &print_to_stream(std::ostream &os) const override;
};

struct GroupByAggResult : public Result {
  Duration group_by_time;
  Duration reduction_time;
  std::vector<Duration> get_reported_timings_list() const override;
  std::ostream &print_to_stream(std::ostream &os) const override;
};

std::ostream &operator<<(std::ostream &os, const Result &res);

struct DwarfRunResult {
  DwarfParams params;
  std::unique_ptr<Result> result;
};

static constexpr auto default_report_header = "host_time_ms,kernel_time_ms";
using SingleRunResults = std::vector<DwarfRunResult>;
class MeasureResults {
public:
  using const_iterator = SingleRunResults::const_iterator;
  MeasureResults(const std::string &name)
      : name_(name), header_(default_report_header) {}

  void add_result(DwarfParams params, std::unique_ptr<Result> result);

  const_iterator begin() const;
  const_iterator end() const;

  void set_report_header(const std::string &header);
  void write_csv(const std::string &filename) const;

private:
  SingleRunResults results_;
  const std::string name_;
  std::string header_;
};
