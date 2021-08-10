#include "cuckoo_hash_build.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"

#include <algorithm>

CuckooHashBuild::CuckooHashBuild() : Dwarf("CuckooHashBuild") {}
const uint32_t EMPTY_KEY = std::numeric_limits<uint32_t>::max();

void CuckooHashBuild::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  
  const std::vector<uint32_t> host_src = /*{0, 4};*/
     helpers::make_unique_random(buf_size);
  
  std::cout << "src: ";
  for (int i = 0; i < buf_size; i++)
    std::cout << host_src[i] << " ";
  std::cout << std::endl;
  const size_t ht_size = buf_size * 2;
  
  //auto sel = get_device_selector(opts);
  cl::sycl::cpu_selector sel;
  sycl::queue q(sel);
  //sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  // std::cout << host_src.size() << "\n";

  for (auto it = 0; it < opts.iterations; ++it) {
      //PolynomialHasher hasher1(ht_size), hasher2(ht_size);

      SimpleHasherWithOffset hasher1(ht_size, (size_t) helpers::make_random()), hasher2(ht_size, (size_t) helpers::make_random());

      std::vector<uint32_t> output(buf_size, 0); //?
      std::vector<uint32_t> expected(buf_size, 1); //?
      size_t bitmask_sz = (ht_size / 32) ? (ht_size / 32) : 1;
      //std::cout << bitmask_sz << " bsz\n";
      std::vector<uint32_t> bitmask(bitmask_sz, 0);
      /*std::array<bool, ht_size> bitmask;
      std::fill_n(out.begin(), ht_size, false);*/
      //std::vector<uint32_t> bitmask(ht_size, 0);
      std::vector<uint32_t> keys(ht_size, EMPTY_KEY);
      std::vector<uint32_t> vals(ht_size, 0);

      sycl::buffer<uint32_t> bitmask_buf(bitmask);
      sycl::buffer<uint32_t> vals_buf(vals);
      sycl::buffer<uint32_t> keys_buf(keys);
      sycl::buffer<uint32_t> src(host_src);
      sycl::buffer<bool, 1>insertion_result_buf{sycl::range{buf_size}}; 

      /*std::cout << "\n" << "iter : " << it << "\n";
      std::cout << host_src.size() << " " << src.get_size() << "\n"
                << bitmask.size() << " " << bitmask_buf.get_size() << "\n"
                << keys.size() << " " << keys_buf.get_size() << "\n"
                << vals.size() << " " << vals_buf.get_size() << "\n"
                << insertion_result_buf.get_size() << "\n";*/
     

      auto host_start = std::chrono::steady_clock::now();
      
      /*while (true)*/ {

        size_t hasher1_offset = (size_t) helpers::make_random();
        size_t hasher2_offset = (size_t) helpers::make_random();
     
        hasher1 = SimpleHasherWithOffset(ht_size, hasher1_offset);
        hasher2 = SimpleHasherWithOffset(ht_size, hasher2_offset);
        std::cout << "hasher offsets: " << hasher1.get_offset() << " " << hasher2.get_offset() << std::endl;
        
        auto clear_keys = q.submit([&](sycl::handler &h) {
          auto keys_acc = keys_buf.get_access(h);
          auto bitmask_acc = bitmask_buf.get_access(h);

          h.parallel_for<class clear_keys>(ht_size, [=](auto &idx) {
            keys_acc[idx] = EMPTY_KEY;
            //bitmask_acc[idx] = 0;
          });
        });

        auto clear_bitmask = q.submit([&](sycl::handler &h) {
          auto bitmask_acc = bitmask_buf.get_access(h);
          h.parallel_for<class clear_bitmask>(bitmask_sz, [=](auto &idx) {
            bitmask_acc[idx] = false;
          });
        });

        q.submit([&](sycl::handler &h) {
          h.depends_on({clear_keys, clear_bitmask});
          auto s = src.get_access(h);
          auto bitmask_acc = bitmask_buf.get_access(h);
          auto keys_acc = keys_buf.get_access(h);
          auto vals_acc = vals_buf.get_access(h);
          auto insertion_acc = insertion_result_buf.get_access(h);
          
          sycl::stream out(10240, 2560, h);

          h.parallel_for<class hash_build>(buf_size, [=](sycl::id<1> idx) {

            CuckooHashtable<uint32_t, uint32_t,  SimpleHasherWithOffset,  SimpleHasherWithOffset> 
            ht(ht_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, out);

            insertion_acc[idx] = ht.insert(s[idx], s[idx], (size_t) idx);
            /*insertion_acc[idx * 5] = ht.insert(s[idx * 5], s[idx * 5], (size_t) idx * 5);
            insertion_acc[idx * 5 + 1] = ht.insert(s[idx * 5 + 1], s[idx * 5 + 1], (size_t) idx * 5 + 1);
            insertion_acc[idx * 5 + 2] = ht.insert(s[idx * 5 + 2], s[idx * 5 + 2], (size_t) idx * 5 + 2);
            insertion_acc[idx * 5 + 3] = ht.insert(s[idx * 5 + 3], s[idx * 5 + 3], (size_t) idx * 5 + 3);
            insertion_acc[idx * 5 + 4] = ht.insert(s[idx * 5 + 4], s[idx * 5 + 4], (size_t) idx * 5 + 4);*/
          });
        }).wait();
        
       auto ht_keys = keys_buf.get_access<sycl::access::mode::read>();
        std::cout << "table : ";
        for (int i = 0; i < ht_size; i++)
           if(ht_keys[i] == EMPTY_KEY)
                  std::cout << "." << " ";
            else std::cout << ht_keys[i] << " ";
        std::cout << std::endl;
        
        auto result = insertion_result_buf.get_access<sycl::access::mode::read>();

        bool pr = false;
        for (int i = 0; i < buf_size; i++){
          if (result[i] == false){
            pr = true;
          }
          std::cout << result[i] << " ";
        }
        std::cout << "\n";
        if (!pr) break;
        //if(std::all_of(result, result + buf_size, [](bool i){ return i;}))
          //break;
      }
      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
      Result result;
      result.host_time = host_end - host_start;
      sycl::buffer<uint32_t> out_buf(output);
      //std::cout << output.size() << " " << out_buf.get_size() << "\n";
      /*(q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto o = out_buf.get_access(h);
       auto bitmask_acc = bitmask_buf.get_access(h);
       auto vals_acc = vals_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);
       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         CuckooHashtable<uint32_t, uint32_t,  SimpleHasherWithOffset,  SimpleHasherWithOffset> 
            ht(ht_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
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
      meter.add_result(std::move(params), std::move(result));*/
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
