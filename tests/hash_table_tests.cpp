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

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access<access::mode::read_write>(h);
    auto data_acc = data_buf.get_access<access::mode::read_write>(h);
    auto keys_acc = keys_buf.get_access<access::mode::read_write>(h);

    h.parallel_for<class test_hash>(range<1>{2}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);
      if (id == 0) {
        ht.insert(2, 2);
        ht.insert(65, 3);
        ht.insert(66, 8);
        ht.insert(1, 9);
      }

      ht.insert(10, id + 1);
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(result[1], 3);
  ASSERT_EQ(result[2], 2);
  ASSERT_EQ(result[3], 8);
  ASSERT_EQ(result[4], 9);
  ASSERT_EQ(result[10] + result[11], 3);
}

TEST(HashTable, Probe) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);
  std::vector<uint32_t> output(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask);
  buffer<uint32_t> data_buf(data);
  buffer<uint32_t> keys_buf(keys);

  buffer<uint32_t> out_buf(output);

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);
    auto out_acc = out_buf.get_access(h);

    h.parallel_for<class test_hash_probe>(range{1}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);

      ht.insert(1, 1);
      ht.insert(1, 5);
      ht.insert(4, 55);

      auto [v1, b1] = ht.at(1);
      auto [v2, b2] = ht.at(4);
      out_acc[idx] = b1 ? v1 : 0;
      out_acc[idx + 1] = b2 ? v2 : 0;
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  auto outer = out_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < input_size; ++i) {
    std::cout << outer[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(outer[0], 1);
  ASSERT_EQ(outer[1], 55);
}

TEST(HashTable, Has) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);
  std::vector<uint32_t> output(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask);
  buffer<uint32_t> data_buf(data);
  buffer<uint32_t> keys_buf(keys);

  buffer<uint32_t> out_buf(output);

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);
    auto out_acc = out_buf.get_access(h);

    h.parallel_for<class test_hash_has>(range{1}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);

      ht.insert(1, 1);
      ht.insert(65, 5);
      ht.insert(1 + 2 * input_size, 6);
      ht.insert(1 + 3 * input_size, 7);
      ht.insert(4, 55);

      out_acc[0] = ht.has(1);
      out_acc[1] = ht.has(65);
      out_acc[2] = ht.has(64);
      out_acc[3] = ht.has(4);
      out_acc[4] = ht.has(129);
      out_acc[5] = ht.has(193);
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  auto outer = out_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < input_size; ++i) {
    std::cout << outer[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(outer[0], 1);
  ASSERT_EQ(outer[1], 1);
  ASSERT_EQ(outer[2], 0);
  ASSERT_EQ(outer[3], 1);
  ASSERT_EQ(outer[4], 1);
  ASSERT_EQ(outer[5], 1);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}