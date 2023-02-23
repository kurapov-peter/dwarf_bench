#include "bench.hpp"

#include "common/registry.hpp"
#include "register_dwarfs.hpp"

#include "common/common.hpp"

#include <cassert>

namespace DwarfBench {

std::string DwarfBench::dwarfToString(DwarfBench::DwarfImpl dwarf, DeviceType device) {
  switch (dwarf) {
  case ConstantExampleDPCPP: {
    return device == DeviceType::CPU ? "ConstantExampleDPCPP" : "ConstantExampleDPCPPCuda";
  }
  case DPLScan: {
    return device == DeviceType::CPU ? "DPLScan" : "DPLScanCuda";
  }
  case GroupBy: {
    return device == DeviceType::CPU ? "GroupBy" : "GroupByCuda";
  }
  case GroupByLocal: {
    assert(device != DeviceType::GPU && "Calling unsupported GroupByLocal for GPU execution dwarf");
    return "GroupByLocal";
  }
  case HashBuild: {
    assert(device != DeviceType::GPU && "Calling unsupported HashBuild for GPU execution dwarf");
    return "HashBuild";
  }
  case HashBuildNonBitmask: {
    assert(device != DeviceType::GPU && "Calling unsupported HashBuildNonBitmask for GPU execution dwarf");
    return "HashBuildNonBitmask";
  }
  case Join: {
    assert(device != DeviceType::GPU && "Calling unsupported Join for GPU execution dwarf");
    return "Join";
  }
  case JoinOmnisci: {
    return device == DeviceType::CPU ? "JoinOmnisci" : "JoinOmnisciCuda";
  }
  case NestedLoopJoin: {
    assert(device != DeviceType::GPU && "Calling unsupported NestedLoopJoin for GPU execution dwarf");
    return "NestedLoopJoin";
  }
  case Radix: {
    return device == DeviceType::CPU ? "Radix" : "RadixCuda";
  }
  case TBBSort: {
    assert(device != DeviceType::GPU && "Calling unsupported TBBSort for GPU execution dwarf");
    return "TBBSort";
  }
  default: {
    return "Unknown Dwarf";
  }
  }
}

std::vector<Measurement> DwarfBench::makeMeasurements(const RunConfig &conf) {
  static Registry *reg = []() {
    populate_registry();
    return Registry::instance();
  }();

  RunOptions _opts = RunOptions{.device_ty = conf.device == DeviceType::CPU
                                                 ? RunOptions::DeviceType::CPU
                                                 : RunOptions::DeviceType::GPU,
                                .input_size = {conf.inputSize},
                                .iterations = conf.iterations,
                                .report_path = ""};

  GroupByRunOptions opts = GroupByRunOptions(_opts, 20, 1024); // TODO

  std::string dwarfName = dwarfToString(dwarfToImpl(conf.dwarf), conf.device);
  auto dwarf = reg->find(dwarfName);
  assert(dwarf != nullptr);

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

DwarfBench::DwarfImpl DwarfBench::dwarfToImpl(Dwarf dwarf) {
  switch (dwarf) {
  case Dwarf::Sort:
    return DwarfImpl::Radix;

  case Dwarf::Join:
    return DwarfImpl::JoinOmnisci;

  case Dwarf::GroupBy:
    return DwarfImpl::GroupBy;

  case Dwarf::Scan:
    return DwarfImpl::DPLScan;
  }

  assert(false); // unreachable
}

DwarfBenchException::DwarfBenchException(const std::string &message)
    : message_(message) {}

const char *DwarfBenchException::what() const noexcept {
  return message_.c_str();
}

} // namespace DwarfBench
