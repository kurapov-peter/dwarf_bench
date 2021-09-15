#include "cuckoo_probe.hpp"
#include "join/join_helpers/join_helpers.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"

CuckooProbe::CuckooProbe() : Dwarf("CuckooProbe") {}
const uint32_t EMPTY_KEY = std::numeric_limits<uint32_t>::max();
const size_t work_groups_count = 256;
constexpr size_t subgroup_size = 32;

void CuckooProbe::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  
  const std::vector<uint32_t> table_a_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values = helpers::make_random<uint32_t>(buf_size);

  const std::vector<uint32_t> table_b_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values = helpers::make_random<uint32_t>(buf_size);

  std::vector<uint32_t> probe_keys(buf_size, 0);
  std::vector<uint32_t> probe_a_values(buf_size, 0);
  std::vector<uint32_t> probe_b_values(buf_size, 0);

  const size_t ht_size = buf_size * 4;
  
  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  size_t work_group_size =
        q.get_device().get_info<sycl::info::device::max_work_group_size>();
  
  sycl::nd_range<1> r{work_group_size * work_groups_count, work_group_size};
 
  for (auto it = 0; it < opts.iterations; ++it) {
      std::unique_ptr<Result> result = std::make_unique<Result>();

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

      sycl::buffer<bool, 1> insertion_result_buf{sycl::range{buf_size}}; 

      sycl::buffer<uint32_t> probe_keys_buf(probe_keys);
      sycl::buffer<uint32_t> probe_a_values_buf(probe_a_values);
      sycl::buffer<uint32_t> probe_b_values_buf(probe_b_values);

      uint32_t scale;
      size_t subgroups_cnt = work_group_size * work_groups_count / subgroup_size;
    
      if (buf_size < subgroups_cnt)
        scale = 1;
      else if (buf_size % (subgroups_cnt) == 0)
        scale = buf_size / subgroups_cnt;
      else
        scale = buf_size / (subgroups_cnt - 1);

      while (true) {
        uint32_t hasher1_seed = helpers::make_random();
        uint32_t hasher2_seed = helpers::make_random();
     
        hasher1 = MurmurHash3_x86_32(ht_size, sizeof(uint32_t), hasher1_seed);
        hasher2 = MurmurHash3_x86_32(ht_size, sizeof(uint32_t), hasher2_seed);

        auto clear_keys = q.submit([&](sycl::handler &h) {
          auto keys_acc = keys_buf.get_access(h);

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

          h.parallel_for<class hash_build>(r, [=](sycl::nd_item<1> it)[[intel::reqd_sub_group_size(subgroup_size)]] {
            CuckooHashtable<uint32_t, uint32_t,  MurmurHash3_x86_32,  MurmurHash3_x86_32> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2);

            int sg_ind = it.get_sub_group().get_local_id();
            if (sg_ind == 0){
              int idx = it.get_global_id() / subgroup_size;
              int end = (idx + 1) * scale;
              if (idx == subgroups_cnt - 1)
                end = buf_size;
              for (int i = idx * scale; i < end && i < buf_size; i++){
                insertion_acc[i] = ht.insert(table_a_keys_acc[i], table_a_values_acc[i]);
              }
            }
          });
        });
        auto insertion_result = insertion_result_buf.get_access<sycl::access::mode::read>();

        bool flag = false;
        for (int i = 0; i < buf_size; i++){
          if (insertion_result[i] == false){
            flag = true;
            break;
          }
        }
        
        if (!flag) {
          auto host_start = std::chrono::steady_clock::now();
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

          auto host_end = std::chrono::steady_clock::now();
          result->host_time = host_end - host_start;
          break;
        }
      }

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

    //   auto expected =
    //     join_helpers::seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);
      
    //   join_helpers::ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = 
    //     {res_keys, {res_a_values, res_b_values}};
      
    //   if (output != expected) {
    //     std::cerr << "Incorrect results" << std::endl;
    //     result->valid = false;
    //   }

      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
  }
}
    
void CuckooProbe::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void CuckooProbe::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
