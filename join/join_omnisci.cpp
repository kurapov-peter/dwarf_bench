
#include "join_omnisci.hpp"
#include "common/dpcpp/omnisci_hashtable.hpp"
#include "join_helpers/join_helpers.hpp"
#include <unordered_set>

size_t count_distinct(const std::vector<uint32_t> &v) {
  std::unordered_set<uint32_t> s{v.begin(), v.end()};
  return s.size();
}

JoinOmnisci::JoinOmnisci() : Dwarf("JoinOmnisci") {}
using namespace join_helpers;
void JoinOmnisci::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  // todo: sizes
  const std::vector<uint32_t> table_a_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_random<uint32_t>(table_a_keys.size());
  size_t unique_keys = count_distinct(table_a_keys);

  const std::vector<uint32_t> table_b_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_random<uint32_t>(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto expected =
      seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);

  const size_t ht_size = buf_size * 2;
  const size_t bitmask_sz = ht_size / 32 + 1;
  SimpleHasher<uint32_t> hasher(ht_size);

  for (unsigned it = 0; it < opts.iterations; ++it) {
    std::unique_ptr<HashJoinResult> result = std::make_unique<HashJoinResult>();
    OmniSci::HashTable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
        table_a_keys, std::numeric_limits<uint32_t>::max(), hasher, ht_size,
        unique_keys, q);
    auto host_start = std::chrono::steady_clock::now();

    ht.build_table();
    ht.build_id_buffer();

    auto build_end = std::chrono::steady_clock::now();

    auto res = ht.lookup(table_b_keys);

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    result->host_time = host_end - host_start;
    result->build_time = build_end - host_start;
    result->probe_time = host_end - build_end;

    std::vector<uint32_t> res_k;
    std::vector<uint32_t> res_va;
    std::vector<uint32_t> res_vb;

    for (int i = 0; i < res.first.size(); i++) {
      res_k.push_back(table_b_keys[res.first[i]]);
      res_vb.push_back(table_b_values[res.first[i]]);
      res_va.push_back(table_a_values[res.second[i]]);
    }
    ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = {res_k,
                                                             {res_va, res_vb}};

    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
    // todo: scale factor?
  }
}

void JoinOmnisci::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void JoinOmnisci::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}