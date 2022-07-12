#include "groupby_global.hpp"
#include "common/dpcpp/hashtable.hpp"
#include <limits>

GroupByGlobal::GroupByGlobal() : GroupBy("Global") {}

void GroupByGlobal::_run(const size_t buf_size, Meter &meter) {
  uint32_t empty_element = _empty_element;
  auto opts = static_cast<const GroupByRunOptions &>(meter.opts());

  const int groups_count = opts.groups_count;
  generate_vals(buf_size);
  generate_keys(buf_size, groups_count);
  generate_expected(groups_count, add);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  size_t hash_size = groups_count * 2;
  PolynomialHasher hasher(hash_size);

  std::vector<uint32_t> ht_vals(hash_size, 0);
  std::vector<uint32_t> ht_keys(hash_size, empty_element);
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

       h.parallel_for<class hash_build>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, PolynomialHasher> ht(
             hash_size, ht_k.get_pointer(), ht_v.get_pointer(), hasher,
             empty_element);

         ht.add(sk[idx], sv[idx]);
       });
     }).wait();

    

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals_buf.get_access(h);
       auto sk = src_keys_buf.get_access(h);
       auto o = out_buf.get_access(h);

       auto ht_v = ht_vals_buf.get_access(h);
       auto ht_k = ht_keys_buf.get_access(h);

       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, PolynomialHasher> ht(
             hash_size, ht_k.get_pointer(), ht_v.get_pointer(), hasher,
             empty_element);

         std::pair<uint32_t, bool> sum_for_group = ht.at(sk[idx]);
         sycl::atomic<uint32_t>(o.get_pointer() + sk[idx])
             .store(sum_for_group.first);
       });
     }).wait();
    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;

    out_buf.get_access<sycl::access::mode::read>();
    result->valid = check_correctness(output);

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));

    q.submit([&](sycl::handler &h) {
       auto o = out_buf.get_access(h);
       auto ht_v = ht_vals_buf.get_access(h);
       auto ht_k = ht_keys_buf.get_access(h);

       h.single_task<class clean>([=]() {
         for (size_t i = 0; i < hash_size; i++) {
           ht_v[i] = 0;
           ht_k[i] = empty_element;
           if (i < groups_count)
            o[i] = 0;
         }
       });
     }).wait();
  }
}
