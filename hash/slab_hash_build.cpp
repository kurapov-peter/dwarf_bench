#include "slab_hash_build.hpp"
#include "common/dpcpp/slab_hash.hpp"

SlabHashBuild::SlabHashBuild() : Dwarf("SlabHashBuild") {}

void SlabHashBuild::_run(const size_t buf_size, Meter &meter) {
  const int scale = 16;                                         //todo how to get through options


  auto opts = meter.opts();
  const std::vector<uint32_t> host_src =
      helpers::make_random<uint32_t>(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  SlabHashHashers::Hasher<32, 48, 1031> hasher;

  for (auto it = 0; it < opts.iterations; ++it) {
    int work_size = (buf_size / scale);
    sycl::nd_range<1> r{SUBGROUP_SIZE * work_size, SUBGROUP_SIZE};
    std::vector<SlabList<pair<uint32_t, uint32_t>>> data(BUCKETS_COUNT);
    for(auto &e: data) {
        e.root = sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>(
                    sycl::malloc_shared<SlabNode<pair<uint32_t, uint32_t>>>(CLUSTER_SIZE, q)
                    );

        for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
            *(e.root + i) = SlabNode<pair<uint32_t, uint32_t>>({EMPTY_UINT32_T, 0});
            (e.root + i)->next = (e.root + i + 1);
        }

    }

    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    {
        sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> data_buf(data);
        sycl::buffer<sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(work_size);
        sycl::buffer<uint32_t> src(host_src);

        auto host_start = std::chrono::steady_clock::now();
        q.submit([&](sycl::handler &h) {
        auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
        auto itrs = sycl::accessor(its, h, sycl::read_write);
        auto s = sycl::accessor(src, h, sycl::read_only);

        h.parallel_for<class slab_hash_build>(r, [=](sycl::nd_item<1> it) {
            size_t ind = it.get_group().get_id();
            SlabHashHashers::Hasher<32, 48, 1031> h;
            SlabHash<uint32_t, uint32_t, SlabHashHashers::Hasher<32, 48, 1031>> ht(EMPTY_UINT32_T, 
                                                                    h, data_acc.get_pointer(), 
                                                                    it, itrs[it.get_group().get_id()]);

            for (int i = ind * scale; i < ind * scale + scale; i++) {
                ht.insert(s[i], s[i]);
            }
            
        });
        }).wait();

        auto host_end = std::chrono::steady_clock::now();
        auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                host_end - host_start)
                                .count();
        Result result;
        result.host_time = host_end - host_start;

        sycl::buffer<uint32_t> out_buf(output);

        q.submit([&](sycl::handler &h) {
        auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
        auto itrs = sycl::accessor(its, h, sycl::read_write);
        auto s = sycl::accessor(src, h, sycl::read_only);
        auto o = sycl::accessor(out_buf, h, sycl::read_write);


        h.parallel_for<class slab_hash_build_check>(r, [=](sycl::nd_item<1> it) {
            size_t ind = it.get_group().get_id();
            SlabHashHashers::Hasher<32, 48, 1031> h;
            SlabHash<uint32_t, uint32_t, SlabHashHashers::Hasher<32, 48, 1031>> ht(EMPTY_UINT32_T, 
                                                                    h, data_acc.get_pointer(), 
                                                                    it, itrs[it.get_group().get_id()]);

            for (int i = ind * scale; i < ind * scale + scale; i++) {
                auto ans = ht.find(s[i]);
                if (it.get_local_id() == 0) o[i] = ans.second;
            }
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

    for (auto &e: data) {
        sycl::free(e.root, q);
    }
  }
}

void SlabHashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SlabHashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
