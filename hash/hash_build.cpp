#include "hash_build.hpp"

#include <cmath>

#include "common/dpcpp/hashtable.hpp"

HashBuild::HashBuild() : Dwarf("HashBuild") {}
void HashBuild::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const std::vector<uint32_t> host_src =
      helpers::make_random<uint32_t>(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  size_t ht_size = buf_size * 2;
  size_t bitmask_sz = std::ceil((float) ht_size / 32);
  MurmurHash3_x86_32 hasher(ht_size, sizeof(uint32_t),
                               helpers::make_random());

  for (auto it = 0; it < opts.iterations; ++it) {
    
    std::vector<uint32_t> bitmask(bitmask_sz, 0);
    std::vector<uint32_t> data(ht_size, 0);
    std::vector<uint32_t> keys(ht_size, 0);
    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src(host_src);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);

       auto bitmask_acc = bitmask_buf.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTable<uint32_t, uint32_t, MurmurHash3_x86_32>
             ht(ht_size, bitmask_sz, keys_acc.get_pointer(), data_acc.get_pointer(),
                bitmask_acc.get_pointer(), hasher);

         ht.insert(s[idx], s[idx]);
       });
     }).wait();

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;

    sycl::buffer<uint32_t> out_buf(output);

    q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto o = out_buf.get_access(h);

       auto bitmask_acc = bitmask_buf.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTable<uint32_t, uint32_t, MurmurHash3_x86_32>
             ht(ht_size, bitmask_sz, keys_acc.get_pointer(), data_acc.get_pointer(),
                bitmask_acc.get_pointer(), hasher);

         o[idx] = ht.has(s[idx]);
       });
     }).wait();

    out_buf.get_access<sycl::access::mode::read>();
    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void HashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void HashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
