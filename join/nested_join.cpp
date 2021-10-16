#include "nested_join.hpp"
#include "common/dpcpp/dpcpp_common.hpp"
#include "join_helpers/join_helpers.hpp"
#include <math.h>

using std::pair;
using namespace join_helpers;
NestedLoopJoin::NestedLoopJoin() : Dwarf("NestedLoopJoin") {}

void NestedLoopJoin::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  const std::vector<uint32_t> table_a_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_random<uint32_t>(table_a_keys.size());

  const std::vector<uint32_t> table_b_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_random<uint32_t>(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<uint32_t> key_out(buf_size * buf_size, 0);
  std::vector<uint32_t> val1_out(buf_size * buf_size, -1);
  std::vector<uint32_t> val2_out(buf_size * buf_size, -1);

  auto expected = join_helpers::seq_join(table_a_keys, table_a_values,
                                         table_b_keys, table_b_values);

  for (auto it = 0; it < opts.iterations; ++it) {
    std::unique_ptr<Result> result = std::make_unique<Result>();

    {
      sycl::buffer<uint32_t> key_a(table_a_keys);
      sycl::buffer<uint32_t> val_a(table_a_values);
      sycl::buffer<uint32_t> key_b(table_b_keys);
      sycl::buffer<uint32_t> val_b(table_b_values);

      sycl::buffer<uint32_t> out_key_b(key_out);
      sycl::buffer<uint32_t> out_val1_b(val1_out);
      sycl::buffer<uint32_t> out_val2_b(val2_out);

      auto host_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto key_a_acc = sycl::accessor(key_a, h, sycl::read_only);
         auto val_a_acc = sycl::accessor(val_a, h, sycl::read_only);

         auto key_b_acc = sycl::accessor(key_b, h, sycl::read_only);
         auto val_b_acc = sycl::accessor(val_b, h, sycl::read_only);

         auto out_key_acc = sycl::accessor(out_key_b, h, sycl::read_write);
         auto out_val1_acc = sycl::accessor(out_val1_b, h, sycl::read_write);
         auto out_val2_acc = sycl::accessor(out_val2_b, h, sycl::read_write);

         h.parallel_for<class nested_join>(buf_size, [=](auto &it) {
           uint32_t key = key_a_acc[it];
           uint32_t val = val_a_acc[it];
           for (int i = 0; i < buf_size; i++) {
             if (key_b_acc[i] == key) {
               out_key_acc[it * buf_size + i] = key;
               out_val1_acc[it * buf_size + i] = val;
               out_val2_acc[it * buf_size + i] = val_b_acc[i];
             }
           }
         });
       }).wait();
      auto host_end = std::chrono::steady_clock::now();

      result->host_time = host_end - host_start;
    }

    std::vector<uint32_t> res_k;
    std::vector<uint32_t> res1;
    std::vector<uint32_t> res2;

    for (int i = 0; i < buf_size * buf_size; i++) {
      if (key_out[i] != ((uint32_t)0)) {
        res_k.push_back(key_out[i]);
        res1.push_back(val1_out[i]);
        res2.push_back(val2_out[i]);
      }
    }

    join_helpers::ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = {
        res_k, {res1, res2}};

    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void NestedLoopJoin::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void NestedLoopJoin::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
