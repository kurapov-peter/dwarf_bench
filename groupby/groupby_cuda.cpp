#include "groupby.hpp"
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

GroupByCuda::GroupByCuda() : Dwarf("GroupByCuda") {}

void GroupByCuda::_run(const size_t buf_size, Meter &meter) {
  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  auto opts = static_cast<const GroupByRunOptions &>(meter.opts());

  const int groups_count = opts.groups_count;
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

  PolynomialHasher hasher(buf_size);

  for (auto it = 0; it < opts.iterations; ++it) {
    std::vector<uint32_t> data(buf_size, 0);
    std::vector<uint32_t> keys(buf_size, empty_element);
    std::vector<uint32_t> output(groups_count, 0);

    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src_vals(host_src_vals);
    sycl::buffer<uint32_t> src_keys(host_src_keys);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto sv = src_vals.get_access(h);
       auto sk = src_keys.get_access(h);

       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, PolynomialHasher> ht(
             buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
             empty_element);

         ht.add(sk[idx], sv[idx]);
       });
     }).wait();

    sycl::buffer<uint32_t> out_buf(output);

    q.submit([&](sycl::handler &h) {
       auto sv = src_vals.get_access(h);
       auto sk = src_keys.get_access(h);
       auto o = out_buf.get_access(h);

       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, PolynomialHasher> ht(
             buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
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

    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void GroupByCuda::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void GroupByCuda::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
