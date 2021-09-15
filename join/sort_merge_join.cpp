#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "sort_merge_join.hpp"
#include "common/dpcpp/dpcpp_common.hpp"
#include "join_helpers/join_helpers.hpp"

SortMergeJoin::SortMergeJoin() : Dwarf("SortMergeJoin") {}

void SortMergeJoin::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  const std::vector<uint32_t> table_a_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values = helpers::make_random<uint32_t>(buf_size);

  std::vector<std::pair<uint32_t, uint32_t>> table_a;
  for (int i = 0; i < buf_size; i++) table_a.push_back({table_a_keys[i], table_a_values[i]});

  const std::vector<uint32_t> table_b_keys = helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values = helpers::make_random<uint32_t>(buf_size);

  std::vector<std::pair<std::uint32_t, uint32_t>> table_b;
  for (int i = 0; i < buf_size; i++) table_b.push_back({table_b_keys[i], table_b_values[i]});

  auto sel = get_device_selector(opts);
  auto dev_policy = oneapi::dpl::execution::device_policy{*sel};
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

    for (auto it = 0; it < opts.iterations; ++it) {

        sycl::buffer<std::pair<uint32_t, uint32_t>> table_a_buf(table_a);
        table_a_buf.set_final_data(nullptr);
        sycl::buffer<std::pair<uint32_t, uint32_t>> table_b_buf(table_b);
        table_b_buf.set_final_data(nullptr);

        std::vector<uint32_t> res_keys;
        std::vector<uint32_t> res_a_values;
        std::vector<uint32_t> res_b_values;


        auto host_start = std::chrono::steady_clock::now();
        std::sort(dev_policy, oneapi::dpl::begin(table_a_buf), oneapi::dpl::end(table_a_buf));
        std::sort(dev_policy, oneapi::dpl::begin(table_b_buf), oneapi::dpl::end(table_b_buf));

        auto table_a_acc = table_a_buf.get_access<sycl::access::mode::read>();
        auto table_b_acc = table_b_buf.get_access<sycl::access::mode::read>();
       
        size_t it_a = 0, it_b = 0;
        while(it_a < buf_size && it_b < buf_size){
            if(table_a_acc[it_a].first == table_b_acc[it_b].first){
                res_keys.push_back(table_a_acc[it_a].first);
                res_a_values.push_back(table_a_acc[it_a].second);
                res_b_values.push_back(table_b_acc[it_b].second);
                it_a++; it_b++;
                continue;
            }
            
            if(table_a_acc[it_a].first < table_b_acc[it_b].first){
                it_a++;
                continue;
            }

            it_b++;
        }

        auto host_end = std::chrono::steady_clock::now();
        std::unique_ptr<Result> result = std::make_unique<Result>();
        result->host_time = host_end - host_start;
        
        // auto expected =
        //     join_helpers::seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);
      
        // join_helpers::ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = 
        //     {res_keys, {res_a_values, res_b_values}};
      
        if (output != expected) {
            std::cerr << "Incorrect results" << std::endl;
            result->valid = false;
        }

        DwarfParams params{{"buf_size", std::to_string(buf_size)}};
        meter.add_result(std::move(params), std::move(result));
    }
}

void SortMergeJoin::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SortMergeJoin::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}