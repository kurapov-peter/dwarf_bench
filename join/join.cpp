#include "join.hpp"

#include "common/dpcpp/hashtable.hpp"

Join::Join() : Dwarf("Join") {}

void Join::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  // todo: sizes
  const std::vector<uint32_t> table_a_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_random<uint32_t>(table_a_keys.size());

  const std::vector<uint32_t> table_b_keys =
      helpers::make_random<uint32_t>(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_random<uint32_t>(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // hash table
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::vector<uint32_t> data(buf_size, 0);
  std::vector<uint32_t> keys(buf_size, 0);

  // testing
  std::vector<uint32_t> output(buf_size, 0);

  sycl::buffer<uint32_t> bitmask_buf(bitmask);
  sycl::buffer<uint32_t> data_buf(data);
  sycl::buffer<uint32_t> keys_buf(keys);

  sycl::buffer<uint32_t> key_a(table_a_keys);
  sycl::buffer<uint32_t> val_a(table_a_values);
  sycl::buffer<uint32_t> key_b(table_b_keys);
  sycl::buffer<uint32_t> val_b(table_b_values);

  auto expected = join_helpers::seq_join(table_a_keys, table_a_values,
                                         table_b_keys, table_b_values);

  // todo: scale factor?
  const size_t ht_size = join_helpers::get_size(expected);
  SimpleHasher<uint32_t> hasher(ht_size);

  auto host_start = std::chrono::steady_clock::now();
  q.submit([&](sycl::handler &h) {
    auto key_a_acc = key_a.get_access(h);
    auto val_a_acc = val_a.get_access(h);

    // ht data accessors
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);

    h.parallel_for<class join_build>(buf_size, [=](auto &idx) {
      SimpleNonOwningHashTable ht(ht_size, keys_acc.get_pointer(),
                                  data_acc.get_pointer(),
                                  bitmask_acc.get_pointer(), hasher);

      // todo: pick smaller one
      ht.insert(key_a_acc[idx], val_a_acc[idx]);
    });
  });

  q.submit([&](sycl::handler &h) {
    auto key_b_acc = key_b.get_access(h);
    auto val_b_acc = val_b.get_access(h);

    // ht data accessors
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);

    h.parallel_for<class join_probe>(buf_size, [=](auto &idx) {
      SimpleNonOwningHashTable ht(ht_size, keys_acc.get_pointer(),
                                  data_acc.get_pointer(),
                                  bitmask_acc.get_pointer(), hasher);
      // todo
    });
  });
}

void Join::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void Join::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
