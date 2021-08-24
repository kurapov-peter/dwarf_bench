#include "cuckoo_hash_build.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"
#include <algorithm>

CuckooHashBuild::CuckooHashBuild() : Dwarf("CuckooHashBuild") {}
const uint32_t EMPTY_KEY = std::numeric_limits<uint32_t>::max();
const uint32_t WORKGROUP_SIZE = 1;
//const size_t MAX_WORKGROUPS_COUNT = 10;
//const uint32_t SCALE = 2;

void CuckooHashBuild::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  
  const std::vector<uint32_t> host_src = 
    helpers::make_unique_random(buf_size);

  const size_t ht_size = buf_size * 4;
  
  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
 
  for (auto it = 0; it < opts.iterations; ++it) {
      std::cout << "here 1"<< std::endl;

      MurmurHash3_x86_32 hasher1(ht_size, sizeof(uint32_t), helpers::make_random()), 
                          hasher2(ht_size, sizeof(uint32_t),helpers::make_random());

      std::vector<uint32_t> output(buf_size, 0);
      std::vector<uint32_t> expected(buf_size, 1);

      size_t bitmask_sz = (ht_size / 32) ? (ht_size / 32) : 1;
      std::vector<uint32_t> bitmask(bitmask_sz, 0);
      std::vector<uint32_t> keys(ht_size, EMPTY_KEY);
      std::vector<uint32_t> vals(ht_size, 0);

      sycl::buffer<uint32_t> bitmask_buf(bitmask);
      sycl::buffer<uint32_t> vals_buf(vals);
      sycl::buffer<uint32_t> keys_buf(keys);
      sycl::buffer<uint32_t> src(host_src);
      sycl::buffer<bool, 1>insertion_result_buf{sycl::range{buf_size}}; 
     

      auto host_start = std::chrono::steady_clock::now();
      uint32_t scale;
      size_t workgroups_cnt = 10;
      if (buf_size % (workgroups_cnt / 2) == 0)
        scale = buf_size / (workgroups_cnt / 2);
      else
        scale = buf_size / (workgroups_cnt / 2 - 1);

      while (true) {
        std::cout << "here 2"<< std::endl;
        uint32_t hasher1_offset = helpers::make_random();
        uint32_t hasher2_offset = helpers::make_random();
     
        hasher1 = MurmurHash3_x86_32(ht_size, sizeof(uint32_t), hasher1_offset);
        hasher2 = MurmurHash3_x86_32(ht_size, sizeof(uint32_t), hasher2_offset);

        auto clear_keys = q.submit([&](sycl::handler &h) {
          auto keys_acc = keys_buf.get_access(h);
          auto bitmask_acc = bitmask_buf.get_access(h);

          h.parallel_for<class clear_keys>(ht_size, [=](auto &idx) {
            keys_acc[idx] = EMPTY_KEY;
          });
        });

        q.submit([&](sycl::handler &h) {
          h.depends_on(clear_keys);
          auto s = src.get_access(h);
          auto bitmask_acc = bitmask_buf.get_access(h);
          auto keys_acc = keys_buf.get_access(h);
          auto vals_acc = vals_buf.get_access(h);
          auto insertion_acc = insertion_result_buf.get_access(h);
          

          h.parallel_for<class hash_build>(sycl::nd_range<1>{workgroups_cnt, WORKGROUP_SIZE}, [=](sycl::nd_item<1> it) {
            CuckooHashtable<uint32_t, uint32_t,  MurmurHash3_x86_32,  MurmurHash3_x86_32> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2);

            size_t idx = it.get_global_id();
              if (idx % 2 == 0){
                idx = idx / 2;
                for (int i = idx * scale; i < (idx + 1) * scale && i < buf_size; i++)
                  insertion_acc[i] = ht.insert(s[i], s[i]);
              }
          });
        }).wait();
        std::cout << "here 3"<< std::endl;
        auto result = insertion_result_buf.get_access<sycl::access::mode::read>();

        bool flag = false;
        for (int i = 0; i < buf_size; i++){
          if (result[i] == false){
            flag = true;
            break;
          }
        }
        if (!flag) break;
      }
      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
      Result result;
      result.host_time = host_end - host_start;
      sycl::buffer<uint32_t> out_buf(output);
      q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto o = out_buf.get_access(h);
       auto bitmask_acc = bitmask_buf.get_access(h);
       auto vals_acc = vals_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);
       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         CuckooHashtable<uint32_t, uint32_t,  MurmurHash3_x86_32,  MurmurHash3_x86_32> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2);
         o[idx] = ht.has(s[idx]);
       });
      }).wait();
      out_buf.get_access<sycl::access::mode::read>();
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result.valid = false;
      }
      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
  }
}
    
void CuckooHashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void CuckooHashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
