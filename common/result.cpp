#include "result.hpp"
#include <fstream>

std::ostream &operator<<(std::ostream &os, const Result &res) {
  os << "Kernel duration: " << ((double)res.kernel_time) / 1000.0 << " us\n"
     << "Host duration:   " << res.host_time.count() << " us\n";
  return os;
}

MeasureResults::const_iterator MeasureResults::begin() const {
  return results_.begin();
}

MeasureResults::const_iterator MeasureResults::end() const {
  return results_.end();
}

void MeasureResults::add_result(DwarfParams params, Result &&result) {
  results_.push_back({params, std::move(result)});
}

void MeasureResults::write_csv(const std::string &filename) const {
  std::ofstream of(filename);
  if (of.is_open()) {
    of << "device_type,buf_size,host_time,kernel_time\n";
    for (const auto &res : results_) {
      of << res.params.at("device_type") << "," << res.params.at("buf_size")
         << ",";
      of << res.result.host_time.count() << "," << res.result.kernel_time
         << "\n";
    }
  } else {
    throw std::runtime_error("Could not open the file at " + filename);
  }
}