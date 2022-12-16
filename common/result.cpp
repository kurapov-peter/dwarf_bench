#include "result.hpp"
#include <fstream>
#include <sstream>

std::ostream &operator<<(std::ostream &os, const Result &res) {
  return res.print_to_stream(os);
}

std::ostream &Result::print_to_stream(std::ostream &os) const {
  os << "Kernel duration: " << kernel_time.count() / 1000.0 << " us\n"
     << "Host duration:   " << host_time.count() << " us\n";

  return os;
}

std::vector<Duration> Result::get_reported_timings_list() const {
  return {host_time, kernel_time};
}

std::ostream &HashJoinResult::print_to_stream(std::ostream &os) const {
  Result::print_to_stream(os);

  os << "Build time: " << build_time.count() << " us\n"
     << "Probe time: " << probe_time.count() << " us\n";

  return os;
}

std::ostream &GroupByAggResult::print_to_stream(std::ostream &os) const {
  Result::print_to_stream(os);

  os << "Group stage time: " << group_by_time.count() << " us\n"
     << "Reduce stage time: " << reduction_time.count() << " us\n";

  return os;
}

std::vector<Duration> GroupByAggResult::get_reported_timings_list() const {
  return {host_time, group_by_time, reduction_time};
}

MeasureResults::const_iterator MeasureResults::begin() const {
  return results_.begin();
}

MeasureResults::const_iterator MeasureResults::end() const {
  return results_.end();
}

void MeasureResults::add_result(DwarfParams params,
                                std::unique_ptr<Result> result) {
  results_.push_back({params, std::move(result)});
}

void MeasureResults::set_report_header(const std::string &header) {
  header_ = header;
}

void MeasureResults::write_csv(const std::string &filename) const {
  bool exists = std::ifstream(filename).good();
  std::ofstream of(filename, std::ios::app);
  if (of.is_open()) {
    if (!exists) {
      of << "device_type,buf_size_bytes," << header_ << "\n";
    }
    for (const auto &res : results_) {
      size_t buf_size_bytes =
          std::stoll(res.params.at("buf_size")) * sizeof(int);
      of << res.params.at("device_type") << "," << buf_size_bytes << ",";
      auto measurements = res.result->get_reported_timings_list();
      // todo: refactor me
      auto transformer = [](const Duration &d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d)
                   .count() /
               1000.0;
      };

      std::ostringstream tmp;
      for (auto &e : measurements) {
        tmp << transformer(e) << ",";
      }
      auto tmp_s = tmp.str();
      if (!tmp_s.empty()) {
        tmp_s.resize(tmp_s.size() - 1);
      }
      of << tmp_s << "\n";
    }
  } else {
    throw std::runtime_error("Could not open the file at " + filename);
  }
}

void MeasureResults::clear() { results_.clear(); }
