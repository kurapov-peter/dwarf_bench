#include "bench.hpp"

#include "common/registry.hpp"
#include "register_dwarfs.hpp"

#include "common/common.hpp"

namespace DwarfBench {

std::string dwarfToString(Dwarf dwarf) {
  switch (dwarf) {
    case CONSTANT_EXAMPLE: {
      return "ConstantExample";
    } break;
    case CONSTANT_EXAMPLE_C_API: {
      return "ConstantExampleCAPI";
    } break;
    case CONSTANT_EXAMPLE_DPCPP: {
      return "ConstantExampleDPCPP";
    } break;
    case CUCKOO_HASH_BUILD: {
      return "CuckooHashBuild";
    } break;
    case DPL_SCAN: {
      return "DPLScan";
    } break;
    case GROUP_BY: {
      return "GroupBy";
    } break;
    case GROUP_BY_LOCAL: {
      return "GroupByLocal";
    } break;
    case HASH_BUILD: {
      return "HashBuild";
    } break;
    case HASH_BUILD_NON_BITMASK: {
      return "HashBuildNonBitmask";
    } break;
    case JOIN: {
      return "Join";
    } break;
    case NESTED_LOOP_JOIN: {
      return "NestedLoopJoin";
    } break;
    case RADIX: {
      return "Radix";
    } break;
    case REDUCE_DPCPP: {
      return "ReduceDPCPP";
    } break;
    case TBB_SORT: {
      return "TBBSort";
    } break;
    case TWO_PASS_SCAN: {
      return "TwoPassScan";
    } break;

    default: {
      return "";
    }
  }
}

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

  std::string dwarfName = dwarfToString(conf.dwarf);
  auto dwarf = reg->find(dwarfName);
  if (dwarf == nullptr) {
    throw DwarfBenchException("Internal error: Wrong dwarf name: `" + dwarfName + "`");
  }
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

DwarfBenchException::DwarfBenchException(const std::string& message) : message_(message) {}

const char* DwarfBenchException::what() const noexcept {
  return message_.c_str();
}

}

