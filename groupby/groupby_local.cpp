#include "groupby_local.hpp"
#include "common/dpcpp/hashtable.hpp"
#include <limits>

GroupByLocal::GroupByLocal() : GroupBy("Local") {}

void GroupByLocal::_run(const size_t buf_size, Meter &meter) {
  uint32_t empty_element = _empty_element;
  auto opts = static_cast<const GroupByRunOptions &>(meter.opts());

  const int groups_count = opts.groups_count;
  const int threads_count = opts.threads_count;
  generate_vals(buf_size);
  generate_keys(buf_size, groups_count);
  generate_expected(groups_count, add);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  size_t hash_size = groups_count * 2;
  SimpleHasher<uint32_t> hasher(hash_size);

  std::vector<uint32_t> ht_vals(hash_size * threads_count, 0);
  std::vector<uint32_t> ht_keys(hash_size * threads_count, empty_element);
  std::vector<uint32_t> output(groups_count, 0);

  sycl::buffer<uint32_t> ht_vals_buf(ht_vals);
  sycl::buffer<uint32_t> ht_keys_buf(ht_keys);
  sycl::buffer<uint32_t> src_vals_buf(src_vals);
  sycl::buffer<uint32_t> src_keys_buf(src_keys);
  sycl::buffer<uint32_t> out_buf(output);

  for (auto it = 0; it < opts.iterations; ++it) {
    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto sv = src_vals_buf.get_access(h);
       auto sk = src_keys_buf.get_access(h);

       auto ht_v = ht_vals_buf.get_access(h);
       auto ht_k = ht_keys_buf.get_access(h);

       h.parallel_for<class groupby_local_hash_build>(
           threads_count, [=](auto &idx) {
             size_t hash_table_ptr_offset = (idx * hash_size);
             auto executor_keys_ptr =
                 ht_k.get_pointer() + hash_table_ptr_offset;
             auto executor_vals_ptr =
                 ht_v.get_pointer() + hash_table_ptr_offset;

             LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
                 hash_size, executor_keys_ptr, executor_vals_ptr, hasher,
                 empty_element);

             for (size_t i = idx; i < buf_size; i += threads_count)
               ht.add(sk[i], sv[i]);
           });
     }).wait();

    auto group_by_end = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals_buf.get_access(h);
       auto sk = src_keys_buf.get_access(h);

       auto ht_v = ht_vals_buf.get_access(h);
       auto ht_k = ht_keys_buf.get_access(h);

       auto o = out_buf.get_access(h);

       h.single_task<class groupby_local_collect>([=]() {
         for (int idx = 0; idx < threads_count; idx++) {
           size_t hash_table_ptr_offset = (idx * hash_size);
           auto executor_keys_ptr = ht_k.get_pointer() + hash_table_ptr_offset;
           auto executor_vals_ptr = ht_v.get_pointer() + hash_table_ptr_offset;

           LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
               hash_size, executor_keys_ptr, executor_vals_ptr, hasher,
               empty_element);

           for (int j = 0; j < groups_count; j++)
             o[j] += ht.at(j).first;
         }
       });
     }).wait();

    auto host_end = std::chrono::steady_clock::now();
    std::unique_ptr<GroupByAggResult> result =
        std::make_unique<GroupByAggResult>();
    result->host_time = host_end - host_start;
    result->group_by_time = group_by_end - host_start;
    result->reduction_time = host_end - group_by_end;

    out_buf.get_access<sycl::access::mode::read>();
    result->valid = check_correctness(output);

    DwarfParams params{{"buf_size", std::to_string(get_size(buf_size))}};
    meter.add_result(std::move(params), std::move(result));

    q.submit([&](sycl::handler &h) {
       auto o = out_buf.get_access(h);
       auto ht_v = ht_vals_buf.get_access(h);
       auto ht_k = ht_keys_buf.get_access(h);

       h.single_task<class clean>([=]() {
         for (size_t i = 0; i < hash_size * threads_count; i++) {
           ht_v[i] = 0;
           ht_k[i] = empty_element;
           if (i < groups_count)
             o[i] = 0;
         }
       });
     }).wait();
  }
}
