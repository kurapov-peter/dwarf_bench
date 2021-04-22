#include "result.hpp"

std::ostream &operator<<(std::ostream &os, const Result &res) {
  os << "Kernel duration: " << res.kernel_time << "us\n"
     << "Host duration: " << res.host_time.count() << "us\n";
  return os;
}

MeasureResults::const_iterator MeasureResults::begin() const {
  return results_.begin();
}

MeasureResults::const_iterator MeasureResults::end() const {
  return results_.end();
}

void MeasureResults::add_result(Result &&result) { results_.push_back(result); }