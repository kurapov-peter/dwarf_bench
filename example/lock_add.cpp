#include <CL/sycl.hpp>

constexpr size_t subgroup_size = 32;

using sycl::access::address_space::global_device_space;
using sycl::ext::oneapi::memory_order::acq_rel;
using sycl::ext::oneapi::memory_scope::device;

template <typename K>
using atomic_ref_device =
    sycl::ext::oneapi::atomic_ref<K, acq_rel, device, global_device_space>;

int main(int argc, char *argv[]) {
  sycl::queue q = std::string(argv[1]).compare("gpu") == 0
                      ? sycl::queue{sycl::gpu_selector{}}
                      : sycl::queue{sycl::cpu_selector{}};

  std::cout
      << q.get_device().get_info<sycl::info::device::max_work_group_size>()
      << std::endl;

  size_t work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();

  size_t work_groups_count = 256;
  uint32_t lock = 0;
  uint32_t num = 0;
  sycl::nd_range<1> r{work_group_size * work_groups_count, work_group_size};

  {
    sycl::buffer<uint32_t> num_buf(&num, sycl::range<1>{1});
    sycl::buffer<uint32_t> lock_buf(&lock, sycl::range<1>{1});

    q.submit([&](sycl::handler &h) {
       auto num_acc = sycl::accessor(num_buf, h, sycl::read_write);
       auto lock_acc = sycl::accessor(lock_buf, h, sycl::read_write);

       h.parallel_for(r, [=
       ](sycl::nd_item<1> it)[[intel::reqd_sub_group_size(subgroup_size)]] {
         int sg_ind = it.get_sub_group().get_local_id();

         if (it.get_sub_group().get_local_id() == 0) {
           bool success = false;

           // lock
           while (!success) {
             uint32_t expected = 0;
             success = atomic_ref_device<uint32_t>(*(lock_acc.get_pointer()))
                           .compare_exchange_weak(expected, 1);
           }

           *(num_acc.get_pointer()) = *(num_acc.get_pointer()) + 1;
           // unlock
           atomic_ref_device<uint32_t>(*(lock_acc.get_pointer())).store(0);
         }
       });
     })
        .wait();
  }
  std::cout << num << '\n';
}

