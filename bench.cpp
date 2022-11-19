#include "bench.hpp"

DwarfBench::DwarfBench() { reg = Registry::instance(); }

std::vector<Measurement> DwarfBench::makeMeasurements(const RunConfig &conf) {
  auto dwarf = reg->find(conf.dwarf);
  dwarf->clear_results();
  dwarf->init(*conf.opts);
  dwarf->run(*conf.opts);

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