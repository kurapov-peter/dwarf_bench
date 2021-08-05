#include "cuckoo_hash_build.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"
CuckooHashBuild::CuckooHashBuild() : Dwarf("CuckooHashBuild") {}
const uint32_t EMPTY_KEY = std::numeric_limits<uint32_t>::max();
void CuckooHashBuild::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const std::vector<uint32_t> host_src =
      helpers::make_unique_random(buf_size);
  const size_t ht_size = buf_size * 2;
  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  for (auto it = 0; it < opts.iterations; ++it) {
      PolynomialHasher hasher1(buf_size * 2), hasher2(buf_size * 2);
      bool sucess_insertion = false;
      std::vector<uint32_t> output(buf_size, 0); //?
      std::vector<uint32_t> expected(buf_size, 1); //?
      size_t bitmask_sz = (ht_size * 2 / 32) ? (ht_size / 32) : 1;
      std::vector<uint32_t> bitmask(bitmask_sz, 0);
      std::vector<uint32_t> keys(ht_size, EMPTY_KEY);
      std::vector<uint32_t> vals(ht_size, 0);
      sycl::buffer<uint32_t> bitmask_buf(bitmask);
      sycl::buffer<uint32_t> vals_buf(vals);
      sycl::buffer<uint32_t> keys_buf(keys);
      sycl::buffer<uint32_t> src(host_src);
      sycl::buffer<bool>sucess_insertion_buf{range{1}}; 
      sycl::host_acesor sucess_insertion_host_acc{sucess_insertion_buf};
      auto host_start = std::chrono::steady_clock::now();
      
      while (true) {
        //sucess_insertion = true;
        hasher1(buf_size * 2);
        hasher2(buf_size * 2);
        sucess_insertion_host_acc[0] = true;
        auto clear_keys = q.submit([&](sycl::handler &h) {
          auto keys_acc = keys_buf.get_access(h);
          h.parallel_for<class clear_keys>(buf_size, [=](auto &idx) {
            keys_acc[idx] = EMPTY_KEY;
          });
        });
        auto clear_bitmask = q.submit([&](sycl::handler &h) {
          auto bitmask_acc = bitmask_buf.get_access(h);
          h.parallel_for<class clear_bitmask>(bitmask_sz, [=](auto &idx) {
            bitmask_acc[idx] = 0;
          });
        });
        q.submit([&](sycl::handler &h) {
          h.depends_on({clear_keys, clear_bitmask});
          auto s = src.get_access(h);
          auto sucess_insertion_acc = sucess_insertion_buf(h);
          auto bitmask_acc = bitmask_buf.get_access(h);
          auto keys_acc = keys_buf.get_access(h);
          auto vals_acc = vals_buf.get_access(h);
          
          h.parallel_for<class hash_build>(buf_size, [=](auto &idx) {
            CuckooHashtable<uint32_t, uint32_t, PolynomialHasher, PolynomialHasher> 
            ht(ht_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, EMPTY_KEY);
            sycl::atomic<bool>(sucess_insertion_buf[0]).fetch_and(ht.insert(s[idx], s[idx]));
          });
        }).wait();
        
        if(sucess_insertion_host_acc[0]) break;
      }
      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
      Result result;
      result.host_time = host_end - host_start;
      sycl::buffer<uint32_t> out_buf(output);
      q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto o = out_buf.get_access(h);
       auto bitmask_acc = bitmask_buf.get_access(h);
       auto vals_acc = vals_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);
       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         CuckooHashtable<uint32_t, uint32_t, PolynomialHasher, PolynomialHasher> 
            ht(ht_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, EMPTY_KEY);
         o[idx] = ht.has(s[idx]);
       });
      }).wait();
      out_buf.get_access<sycl::access::mode::read>();
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result.valid = false;
      }
      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
  }
}
    
void CuckooHashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void CuckooHashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
