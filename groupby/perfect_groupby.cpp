#include "perfect_groupby.hpp"
#include "common/dpcpp/dpcpp_common.hpp"
#include "common/dpcpp/perfect_hashtable.hpp"
#include <limits>

PerfectGroupBy::PerfectGroupBy() : GroupBy("Perfect") {}

void PerfectGroupBy::_run(const size_t buf_size, Meter &meter) {
  uint32_t empty_element = _empty_element;
  auto opts = static_cast<const GroupByRunOptions &>(meter.opts());

  const int groups_count = opts.groups_count;
  size_t threads_count = opts.threads_count;
  size_t work_group_size = opts.work_group_size;
  generate_vals(buf_size);
  generate_keys(buf_size, groups_count);
  generate_expected(groups_count, add);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  const size_t min_key = 0;
  std::vector<uint32_t> ht_vals(groups_count * threads_count, 0);
  std::vector<uint32_t> output(groups_count, 0);

  sycl::buffer<uint32_t> ht_vals_buf(ht_vals);
  sycl::buffer<uint32_t> src_vals_buf(src_vals);
  sycl::buffer<uint32_t> src_keys_buf(src_keys);
  sycl::buffer<uint32_t> out_buf(output);

  for (auto it = 0; it < opts.iterations; ++it) {
    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto sv = src_vals_buf.get_access(h);
       auto sk = src_keys_buf.get_access(h);

       auto ht_v = ht_vals_buf.get_access(h);

       h.parallel_for<class groupby_local_hash_build>(
           sycl::nd_range{{threads_count}, {work_group_size}}, [=](auto &idx) {
             PerfectHashTable<uint32_t, uint32_t> ht(
                 groups_count,
                 ht_v.get_pointer() + idx.get_global_id() * groups_count,
                 min_key);

             for (size_t i = idx.get_global_id(); i < buf_size;
                  i += threads_count)
               ht.add(sk[i], sv[i]);
           });
     }).wait();

    auto group_by_end = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals_buf.get_access(h);
       auto sk = src_keys_buf.get_access(h);

       auto ht_v = ht_vals_buf.get_access(h);

       auto o = out_buf.get_access(h);

       h.single_task<class groupby_local_collect>([=]() {
         for (size_t idx = 0; idx < threads_count; idx++) {
           PerfectHashTable<uint32_t, uint32_t> ht(
               groups_count, ht_v.get_pointer() + idx * groups_count, min_key);

           for (int j = 0; j < groups_count; j++)
             o[j] += ht.at(j);
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

       h.single_task<class clean>([=]() {
         for (size_t i = 0; i < groups_count * threads_count; i++) {
           ht_v[i] = 0;
           o[i] = 0;
         }
       });
     }).wait();
  }
}

size_t PerfectGroupBy::get_size(size_t buf_size) { return buf_size; }