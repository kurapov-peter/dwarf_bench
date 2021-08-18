#include "cuckoo_join.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"

CuckooJoin::CuckooJoin() : Dwarf("CuckooJoin") {}
const uint32_t EMPTY_KEY = std::numeric_limits<uint32_t>::max();
const uint32_t WORKGROUP_SIZE = 1;
const uint32_t SCALE = 2;

void CuckooJoin::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  
  const std::vector<uint32_t> table_a_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values = helpers::make_random<uint32_t>(buf_size);

  // std::cout << "table a" << std::endl;
  // for (int i = 0; i < buf_size; i++)
  //   std::cout << table_a_keys[i] << " " << table_a_values[i] << std::endl;
  // std::cout << std::endl;

  const std::vector<uint32_t> table_b_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values = helpers::make_random<uint32_t>(buf_size);

  // std::cout << "table b" << std::endl;
  // for (int i = 0; i < buf_size; i++)
  //   std::cout << table_b_keys[i] << " " << table_b_values[i] << std::endl;
  // std::cout << std::endl;
  
  std::vector<uint32_t> probe_keys(buf_size, 0);
  std::vector<uint32_t> probe_a_values(buf_size, 0);
  std::vector<uint32_t> probe_b_values(buf_size, 0);

  const size_t ht_size = buf_size * 4;
  
  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
 
  for (auto it = 0; it < opts.iterations; ++it) {

      MurmurHash3_x86_32 hasher1(ht_size, sizeof(uint32_t), helpers::make_random()), 
                          hasher2(ht_size, sizeof(uint32_t),helpers::make_random());

      size_t bitmask_sz = (ht_size / 32) ? (ht_size / 32) : 1;
      std::vector<uint32_t> bitmask(bitmask_sz, 0);
      std::vector<uint32_t> keys(ht_size, EMPTY_KEY);
      std::vector<uint32_t> vals(ht_size, 0);

      // hashtable
      sycl::buffer<uint32_t> bitmask_buf(bitmask);
      sycl::buffer<uint32_t> vals_buf(vals);
      sycl::buffer<uint32_t> keys_buf(keys);

      // data
      sycl::buffer<uint32_t> table_a_keys_buf(table_a_keys);
      sycl::buffer<uint32_t> table_a_values_buf(table_a_values);

      sycl::buffer<uint32_t> table_b_keys_buf(table_b_keys);
      sycl::buffer<uint32_t> table_b_values_buf(table_b_values);

      sycl::buffer<bool, 1>insertion_result_buf{sycl::range{buf_size}}; 

      sycl::buffer<uint32_t> probe_keys_buf(probe_keys);
      sycl::buffer<uint32_t> probe_a_values_buf(probe_a_values);
      sycl::buffer<uint32_t> probe_b_values_buf(probe_b_values);

     

      auto host_start = std::chrono::steady_clock::now();
      
      while (true) {
        std::cout << "here1\n";
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

        auto join_build = q.submit([&](sycl::handler &h) {
          h.depends_on(clear_keys);
          auto table_a_keys_acc = table_a_keys_buf.get_access(h);
          auto table_a_values_acc = table_a_values_buf.get_access(h);

          auto bitmask_acc = bitmask_buf.get_access(h);
          auto keys_acc = keys_buf.get_access(h);
          auto vals_acc = vals_buf.get_access(h);
          auto insertion_acc = insertion_result_buf.get_access(h);

          h.parallel_for<class join_build>(sycl::nd_range<1>{buf_size, WORKGROUP_SIZE}, [=](sycl::nd_item<1> it) {
            CuckooHashtable<uint32_t, uint32_t,  MurmurHash3_x86_32,  MurmurHash3_x86_32> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2);

            size_t idx = it.get_global_id();
            if (idx % 2 == 0){
              for (int i = idx; i < idx + SCALE && i < buf_size; i++)
              insertion_acc[i] = ht.insert(table_a_keys_acc[i], table_a_values_acc[i]);
            }
          });
        });
        std::cout << "here2\n";
        auto result = insertion_result_buf.get_access<sycl::access::mode::read>();

        bool flag = false;
        for (int i = 0; i < buf_size; i++){
          if (result[i] == false){
            flag = true;
            break;
          }
        }
        
        if (!flag) {
          q.submit([&](sycl::handler &h) {
            h.depends_on(join_build);
          
            auto table_a_keys_acc = table_a_keys_buf.get_access(h);
            auto table_a_values_acc = table_a_values_buf.get_access(h);

            auto table_b_keys_acc = table_b_keys_buf.get_access(h);
            auto table_b_values_acc = table_b_values_buf.get_access(h);

            auto probe_keys_acc = probe_keys_buf.get_access(h);
            auto probe_a_values_acc = probe_a_values_buf.get_access(h);
            auto probe_b_values_acc = probe_b_values_buf.get_access(h);

            auto bitmask_acc = bitmask_buf.get_access(h);
            auto keys_acc = keys_buf.get_access(h);
            auto vals_acc = vals_buf.get_access(h);

            h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
              
              CuckooHashtable<uint32_t, uint32_t,  MurmurHash3_x86_32,  MurmurHash3_x86_32> 
                    ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                          bitmask_acc.get_pointer(), hasher1, hasher2);
              
              auto r = ht.at(table_b_keys_acc[idx]);
              if (r.second){
                probe_keys_acc[idx] = table_b_keys_acc[idx];
                probe_a_values_acc[idx] = r.first;
                probe_b_values_acc[idx] = table_b_values_acc[idx];
              }
            });

          }).wait();
          std::cout << "here3\n";
          break;
        }
      }

      auto host_end = std::chrono::steady_clock::now();

      
      //auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
      Result result;
      result.host_time = host_end - host_start;

      std::vector<uint32_t> res_keys;
      std::vector<uint32_t> res_a_values;
      std::vector<uint32_t> res_b_values;

      auto probe_keys_acc = probe_keys_buf.get_access<sycl::access::mode::read>();
      auto probe_a_values_acc = probe_a_values_buf.get_access<sycl::access::mode::read>();
      auto probe_b_values_acc = probe_b_values_buf.get_access<sycl::access::mode::read>();

      for (int i = 0; i < buf_size; i++){
        if (probe_keys_acc[i] != 0){
          res_keys.push_back(probe_keys_acc[i]);
          res_a_values.push_back(probe_a_values_acc[i]);
          res_b_values.push_back(probe_b_values_acc[i]);
        }
      }

      auto expected =
        join_helpers::seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);
      
      join_helpers::ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = 
        {res_keys, {res_a_values, res_b_values}};

      // std::cout << "EXPECTED" << std::endl;

      // for (int i = 0; i < expected.first.size(); i++)
      //   std::cout << expected.first[i] << " : " << expected.second.first[i] << " " << expected.second.second[i] << std::endl;

      // std::cout << "\nOUTPUT" << std::endl;

      // for (int i = 0; i < output.first.size(); i++)
      //   std::cout << output.first[i] << " : " << output.second.first[i] << " " << output.second.second[i] << std::endl;
         
      
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result.valid = false;
      }

      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
  }
}
    
void CuckooJoin::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void CuckooJoin::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
