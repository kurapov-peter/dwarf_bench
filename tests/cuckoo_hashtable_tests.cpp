#include "common/dpcpp/cuckoo_hashtable.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace CuckooTest{
    template <size_t Size> struct Hasher1 {
        size_t operator()(const uint32_t &v) const { return v % Size; }
    };
    
    template <size_t Size> struct Hasher2 {
        size_t operator()(const uint32_t &v) const { return (v % Size + 1) % Size; }
    };
}

#define EMPTY_KEY 2147483647

TEST(CuckooHashTable, insert) {
  const size_t buf_size = 10;
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  
  CuckooTest::Hasher1<buf_size> hasher1;
  CuckooTest::Hasher2<buf_size> hasher2;

  sycl::cpu_selector device_selector;
  sycl::queue q(device_selector);

  std::vector<uint32_t> keys(buf_size, EMPTY_KEY);
  std::vector<uint32_t> vals(buf_size, 0);
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  
  {
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> vals_buf(vals);
    sycl::buffer<uint32_t> bitmask_buf(bitmask);

    q.submit([&](sycl::handler &h) {
      auto keys_acc = keys_buf.get_access(h);
      auto vals_acc = vals_buf.get_access(h);
      auto bitmask_acc = bitmask_buf.get_access(h);

      h.single_task([=]() {
        CuckooHashtable<uint32_t, uint32_t, CuckooTest::Hasher1<buf_size>, CuckooTest::Hasher2<buf_size>> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, EMPTY_KEY);
        
        ht.insert(5, 5, 0);
        ht.insert(15, 15, 0);
        ht.insert(2, 2, 0);
      });
    });

  }

  for (int i = 0; i < buf_size; i++)
    std::cout << keys[i] << " " << vals[i] << "\n";
  
  ASSERT_EQ(keys[5], 5);
  ASSERT_EQ(keys[6], 15);
  ASSERT_EQ(keys[2], 2);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
