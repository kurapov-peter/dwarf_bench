#include "groupby_local.hpp"
#include "common/dpcpp/hashtable.hpp"
#include <limits>

GroupByLocal::GroupByLocal() : Dwarf("GroupByLocal") {}
void GroupByLocal::_run(const size_t buf_size, Meter &meter) {
  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  auto opts = meter.opts();

  const int groups_count = opts.groups_count;
  const int executors = opts.executors;
  const std::vector<uint32_t> host_src_vals =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> host_src_keys =
      helpers::make_random<uint32_t>(buf_size, 0, groups_count - 1);

  std::vector<uint32_t> expected(groups_count);
  for (int i = 0; i < buf_size; i++) {
    expected[host_src_keys[i]] += host_src_vals[i];
  }

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

       auto o = out_buf.get_access(h);

       h.parallel_for<class hash_build>(executors, [=](auto &idx) {
         LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
             groups_count,
             keys_acc.get_pointer() + ((size_t)idx * groups_count),
             data_acc.get_pointer() + ((size_t)idx * groups_count), hasher,
             empty_element);
         for (size_t i = buf_size / executors * idx;
              i < buf_size / executors * (idx + 1); i++)
           ht.add(sk[i], sv[i]);
       });
     }).wait();

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals.get_access(h);
       auto sk = src_keys.get_access(h);

       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       auto o = out_buf.get_access(h);

       h.single_task<class groupby_local_collect>([=]() {
         for (int i = 0; i < executors; i++) {
           LinearHashtable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
               groups_count,
               keys_acc.get_pointer() + ((size_t)i * groups_count),
               data_acc.get_pointer() + ((size_t)i * groups_count), hasher,
               empty_element);
           for (int j = 0; j < groups_count; j++) {
             o[j] += ht.at(j).first;
           }
         }
       });
     }).wait();

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;
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
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
