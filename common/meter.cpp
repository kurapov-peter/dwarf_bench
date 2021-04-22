#include "meter.hpp"

void Meter::add_result(Result &&result) {
  result_.add_result(std::move(result));
}
