
#include "join.hpp"
#include "common/dpcpp/hashtable.hpp"
#include "join_helpers/join_helpers.hpp"

Join::Join() : Dwarf("Join") {}
using namespace join_helpers;
void Join::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  // todo: sizes
  const std::vector<uint32_t> table_a_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_unique_random(table_a_keys.size());

  const std::vector<uint32_t> table_b_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_unique_random(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // hash table
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::vector<uint32_t> data(buf_size, 0);
  std::vector<uint32_t> keys(buf_size, 0);

  // testing
  std::vector<uint32_t> key_out(buf_size, -1);
  std::vector<uint32_t> val1_out(buf_size, -1);
  std::vector<uint32_t> val2_out(buf_size, -1);

  auto expected =
      seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);

  std::unique_ptr<HashJoinResult> result = std::make_unique<HashJoinResult>();

  {

    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);

    sycl::buffer<uint32_t> key_a(table_a_keys);
    sycl::buffer<uint32_t> val_a(table_a_values);
    sycl::buffer<uint32_t> key_b(table_b_keys);
    sycl::buffer<uint32_t> val_b(table_b_values);

    sycl::buffer<uint32_t> out_key_b(key_out);
    sycl::buffer<uint32_t> out_val1_b(val1_out);
    sycl::buffer<uint32_t> out_val2_b(val2_out);

    const size_t ht_size = buf_size;
    SimpleHasher<uint32_t> hasher(ht_size);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto key_a_acc = key_a.get_access(h);
       auto val_a_acc = val_a.get_access(h);

       // ht data accessors
       auto bitmask_acc = bitmask_buf.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class join_build>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTable ht(ht_size, keys_acc.get_pointer(),
                                     data_acc.get_pointer(),
                                     bitmask_acc.get_pointer(), hasher);

         // todo: insert
       });
       q.wait();
     }).wait();
    auto build_end = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
       auto key_b_acc = key_b.get_access(h);
       auto val_b_acc = val_b.get_access(h);

       auto out_key_a = out_key_b.get_access(h);
       auto out_val1_a = out_val1_b.get_access(h);
       auto out_val2_a = out_val2_b.get_access(h);

       // ht data accessors
       auto bitmask_acc = bitmask_buf.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class join_probe>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTable ht(ht_size, keys_acc.get_pointer(),
                                     data_acc.get_pointer(),
                                     bitmask_acc.get_pointer(), hasher);
         auto ans = ht.at(key_b_acc[idx]);
         if (ans.second) {
           out_key_a[idx] = key_b_acc[idx];
           out_val1_a[idx] = ans.first;
           out_val2_a[idx] = val_b_acc[idx];
         }
       });
     }).wait();
    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    result->host_time = host_end - host_start;
    result->build_time = build_end - host_start;
    result->probe_time = host_end - build_end;
  }

  std::vector<uint32_t> res_k;
  std::vector<uint32_t> res1;
  std::vector<uint32_t> res2;

  for (int i = 0; i < buf_size; i++) {
    if (key_out[i] != ((uint32_t)-1)) {
      res_k.push_back(key_out[i]);
      res1.push_back(val1_out[i]);
      res2.push_back(val2_out[i]);
    }
  }

  ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = {res_k, {res1, res2}};

  if (output != expected) {
    std::cerr << "Incorrect results" << std::endl;
    result->valid = false;
  }

  DwarfParams params{{"buf_size", std::to_string(buf_size)}};
  meter.add_result(std::move(params), std::move(result));
  // todo: scale factor?
}

void Join::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void Join::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}