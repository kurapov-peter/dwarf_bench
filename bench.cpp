#include "bench.hpp"

#include "common/registry.hpp"
#include "register_dwarfs.hpp"

#include "common/common.hpp"

namespace DwarfBench {

std::vector<Measurement> DwarfBench::makeMeasurements(const RunConfig &conf) {
  static Registry *reg = []() {
    populate_registry();
    return Registry::instance();
  }();

  RunOptions _opts = RunOptions {
    .device_ty = conf.device == DeviceType::CPU ? RunOptions::DeviceType::CPU : RunOptions::DeviceType::GPU,
    .input_size = { conf.inputSize },
    .iterations = conf.iterations,
    .report_path = ""
  };

  GroupByRunOptions opts = GroupByRunOptions(_opts, 20, 1024);   // TODO

  auto dwarf = reg->find(conf.dwarf);
  dwarf->clear_results();
  dwarf->init(opts);
  dwarf->run(opts);

  std::vector<Measurement> ms;

  std::for_each(
      dwarf->get_results().begin(), dwarf->get_results().end(),
      [&ms](const DwarfRunResult &res) {
        Measurement m = {
            .dataSize = (size_t)std::stoi(
                res.params.at("buf_size")), // todo make bytes counting
            .microseconds = (size_t)res.result->host_time.count() // todo
        };

        ms.push_back(m);
      });

  return ms;
}

}