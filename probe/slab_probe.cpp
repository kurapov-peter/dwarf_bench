#include "common/dpcpp/slab_hash.hpp"
#include "slab_probe.hpp"
#include <math.h>

using std::pair;

SlabProbe::SlabProbe() : Dwarf("SlabProbe") {}

void SlabProbe::_run(const size_t buf_size, Meter &meter) {
  const int scale = 16; // todo how to get through options

  auto opts = meter.opts();
  const std::vector<uint32_t> host_src = helpers::make_unique_random(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  for (auto it = 0; it < opts.iterations; ++it) {
    int work_size = ceil((float)buf_size / scale);

    sycl::nd_range<1> r{SlabHash::SUBGROUP_SIZE * work_size,
                        SlabHash::SUBGROUP_SIZE};
    SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);

    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    {
      sycl::buffer<SlabHash::SlabList<pair<uint32_t, uint32_t>>> data_buf(
          adap._data);
      sycl::buffer<uint32_t> lock_buf(adap._lock);
      sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>> heap_buf(
          &adap._heap, sycl::range<1>{1});
      sycl::buffer<
          sycl::device_ptr<SlabHash::SlabNode<pair<uint32_t, uint32_t>>>>
          its(work_size);
      sycl::buffer<uint32_t> src(host_src);

      q.submit([&](sycl::handler &h) {
         auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
         auto itrs = sycl::accessor(its, h, sycl::read_write);
         auto s = sycl::accessor(src, h, sycl::read_only);
         auto heap_acc = sycl::accessor(heap_buf, h, sycl::read_write);
         auto lock_acc = sycl::accessor(lock_buf, h, sycl::read_write);

         h.parallel_for<class slab_hash_build>(r, [=](sycl::nd_item<1> it) {
           size_t ind = it.get_group().get_id();

           SlabHash::DefaultHasher<5, 11, 1031> h;
           SlabHash::SlabHashTable<uint32_t, uint32_t,
                                   SlabHash::DefaultHasher<5, 11, 1031>>
               ht(SlabHash::EMPTY_UINT32_T, h, data_acc.get_pointer(), it,
                  itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                  *heap_acc.get_pointer());

           for (int i = ind * scale; i < ind * scale + scale && i < buf_size;
                i++) {
             ht.insert(s[i], s[i]);
           }
         });
       }).wait();

      sycl::buffer<uint32_t> out_buf(output);
      auto host_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
         auto itrs = sycl::accessor(its, h, sycl::read_write);
         auto s = sycl::accessor(src, h, sycl::read_only);
         auto o = sycl::accessor(out_buf, h, sycl::read_write);
         auto heap_acc = sycl::accessor(heap_buf, h, sycl::read_write);
         auto lock_acc = sycl::accessor(lock_buf, h, sycl::read_write);

         h.parallel_for<class slab_hash_build_check>(
             r, [=](sycl::nd_item<1> it) {
               size_t ind = it.get_group().get_id();
               SlabHash::DefaultHasher<5, 11, 1031> h;
               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<5, 11, 1031>>
                   ht(SlabHash::EMPTY_UINT32_T, h, data_acc.get_pointer(), it,
                      itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                      *heap_acc.get_pointer());

               for (int i = ind * scale;
                    i < ind * scale + scale && i < buf_size; i++) {
                 auto ans = ht.find(s[i]);
                 if (it.get_local_id() == 0) {
                   o[i] = static_cast<bool>(ans);
                 }
               }
             });
       }).wait();

      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time =
          std::chrono::duration_cast<std::chrono::microseconds>(host_end -
                                                                host_start)
              .count();
      Result result;
      result.host_time = host_end - host_start;

      out_buf.get_access<sycl::access::mode::read>();
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result.valid = false;
      }

      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
    }
  }
}

void SlabProbe::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SlabProbe::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
