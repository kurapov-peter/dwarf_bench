#include "common/dpcpp/hashtable.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(HashTable, Build) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask.data(), bitmask.size());
  buffer<uint32_t> data_buf(data.data(), data.size());
  buffer<uint32_t> keys_buf(keys.data(), keys.size());

  SimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access<access::mode::read_write>(h);
    auto data_acc = data_buf.get_access<access::mode::read_write>(h);
    auto keys_acc = keys_buf.get_access<access::mode::read_write>(h);

    h.parallel_for<class test_hash>(range<1>{1}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t, SimpleHasher<input_size>> ht(
          input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
          bitmask_acc.get_pointer(), hasher);
      ht.insert(2, 2);
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(result[2], 2);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}