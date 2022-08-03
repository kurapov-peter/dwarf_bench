#include "groupby_local.hpp"
#include "common/dpcpp/hashtable.hpp"
#include <limits>

namespace {
using Func = std::function<uint32_t(uint32_t, uint32_t)>;

std::vector<uint32_t> expected_GroupBy(const std::vector<uint32_t> &keys,
                                       const std::vector<uint32_t> &vals,
                                       size_t groups_count, Func f) {
  std::vector<uint32_t> result(groups_count);
  size_t data_size = keys.size();

  for (int i = 0; i < data_size; i++) {
    result[keys[i]] = f(result[keys[i]], vals[i]);
  }

  return result;
}
} // namespace

GroupByLocal::GroupByLocal() : Dwarf("GroupByLocal") {}

void GroupByLocal::_run(const size_t buf_size, Meter &meter) {
  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  auto opts = static_cast<const GroupByRunOptions &>(meter.opts());

  const int groups_count = opts.groups_count;
  const int executors = opts.executors;
  const std::vector<uint32_t> host_src_vals =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> host_src_keys =
      helpers::make_random<uint32_t>(buf_size, 0, groups_count - 1);

  std::vector<uint32_t> expected =
      expected_GroupBy(host_src_keys, host_src_vals, groups_count,
                       [](uint32_t x, uint32_t y) { return x + y; });

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  SimpleHasher<uint32_t> hasher(groups_count);

  for (auto it = 0; it < opts.iterations; ++it) {
    std::vector<uint32_t> data(groups_count * executors, 0);
    std::vector<uint32_t> keys(groups_count * executors, empty_element);
    std::vector<uint32_t> output(groups_count, 0);

    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src_vals(host_src_vals);
    sycl::buffer<uint32_t> src_keys(host_src_keys);
    sycl::buffer<uint32_t> out_buf(output);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto sv = src_vals.get_access(h);
       auto sk = src_keys.get_access(h);

       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       const size_t work_per_executor = std::ceil((float)buf_size / executors);

       h.parallel_for<class groupby_local_hash_build>(
           executors, [=](auto &idx) {
             size_t hash_table_ptr_offset = (idx * groups_count);
             auto executor_keys_ptr =
                 keys_acc.get_pointer() + hash_table_ptr_offset;
             auto executor_vals_ptr =
                 data_acc.get_pointer() + hash_table_ptr_offset;

             LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
                 groups_count, executor_keys_ptr, executor_vals_ptr, hasher,
                 empty_element);

             for (size_t i = work_per_executor * idx;
                  i < work_per_executor * (idx + 1) && i < buf_size; i++)
               ht.add(sk[i], sv[i]);
           });
     }).wait();

    auto group_by_end = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals.get_access(h);
       auto sk = src_keys.get_access(h);

       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       auto o = out_buf.get_access(h);

       h.single_task<class groupby_local_collect>([=]() {
         for (int idx = 0; idx < executors; idx++) {
           size_t hash_table_ptr_offset = (idx * groups_count);
           auto executor_keys_ptr =
               keys_acc.get_pointer() + hash_table_ptr_offset;
           auto executor_vals_ptr =
               data_acc.get_pointer() + hash_table_ptr_offset;

           LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
               groups_count, executor_keys_ptr, executor_vals_ptr, hasher,
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

    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void GroupByLocal::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void GroupByLocal::init(const RunOptions &opts) {
  reporting_header_ = "total_time,group_by_time,reduction_time";
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
