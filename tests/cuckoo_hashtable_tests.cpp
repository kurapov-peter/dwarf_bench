#include "common/dpcpp/cuckoo_hashtable.hpp"
#include <gtest/gtest.h>
#include <vector>

// namespace CuckooTest{
//     template <size_t Size> struct Hasher1 {
//         size_t operator()(const uint32_t &v) const { return v % Size; }
//     };
    
//     template <size_t Size> struct Hasher2 {
//         size_t operator()(const uint32_t &v) const { return (v % Size + 1) % Size; }
//     };
// }

#define EMPTY_KEY 2147483647

TEST(CuckooHashTable, insert) {
  const size_t buf_size = 10;
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  
  StaticSimpleHasher<buf_size> hasher1;
  StaticSimpleHasherWithOffset<buf_size, 1> hasher2;

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
        CuckooHashtable<uint32_t, uint32_t, StaticSimpleHasher<buf_size>, StaticSimpleHasherWithOffset<buf_size, 1>> 
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

TEST(CuckooHashtable, at) {
  const size_t buf_size = 10;
  const size_t output_size = 7;
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  
  StaticSimpleHasher<buf_size> hasher1;
  StaticSimpleHasherWithOffset<buf_size, 1> hasher2;

  sycl::cpu_selector device_selector;
  sycl::queue q(device_selector);

  std::vector<uint32_t> keys(buf_size, EMPTY_KEY);
  std::vector<uint32_t> vals(buf_size, 0);
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::vector<std::pair<uint32_t, bool>> out(output_size, {0, false});
  
  {
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> vals_buf(vals);
    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<std::pair<uint32_t, bool>> out_buf(out);

    q.submit([&](sycl::handler &h) {
      auto keys_acc = keys_buf.get_access(h);
      auto vals_acc = vals_buf.get_access(h);
      auto bitmask_acc = bitmask_buf.get_access(h);
      auto out_acc = out_buf.get_access(h);

      h.single_task([=]() {
        CuckooHashtable<uint32_t, uint32_t,  StaticSimpleHasher<buf_size>, StaticSimpleHasherWithOffset<buf_size, 1>> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, EMPTY_KEY);
        
        ht.insert(5, 5, 0);
        ht.insert(15, 15, 0);
        ht.insert(2, 2, 0);
        ht.insert(258, 258, 0);
        ht.insert(6, 6, 0);

        out_acc[0] = ht.at(5);
        out_acc[1] = ht.at(4);
        out_acc[2] = ht.at(15);
        out_acc[3] = ht.at(2);
        out_acc[4] = ht.at(258);
        out_acc[5] = ht.at(259);
        out_acc[6] = ht.at(6);

      });
    });

  }

  for (int i = 0; i < buf_size; i++)
    std::cout << keys[i] << " " << vals[i] << "\n";
  
  ASSERT_EQ(out[0].first, 5);
  ASSERT_EQ(out[1].second, false);
  ASSERT_EQ(out[2].first, 15);
  ASSERT_EQ(out[3].first, 2);
  ASSERT_EQ(out[4].first, 258);
  ASSERT_EQ(out[5].second, false);
  ASSERT_EQ(out[6].first, 6);
}

TEST(CuckooHashtable, fails_to_insert) {
  const size_t buf_size = 5;
  const size_t output_size = 4;
  size_t bitmask_sz = (buf_size / 32) ? (buf_size / 32) : 1;
  
  StaticSimpleHasher<buf_size> hasher1;
  StaticSimpleHasherWithOffset<buf_size, 1> hasher2;

  sycl::cpu_selector device_selector;
  sycl::queue q(device_selector);

  std::vector<uint32_t> keys(buf_size, EMPTY_KEY);
  std::vector<uint32_t> vals(buf_size, 0);
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::array<bool, output_size> out;
  std::fill_n(out.begin(), output_size, false);
  
  //std::vector<bool> out(output_size, false);
  
  {
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> vals_buf(vals);
    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<bool> out_buf(out);

    q.submit([&](sycl::handler &h) {
      auto keys_acc = keys_buf.get_access(h);
      auto vals_acc = vals_buf.get_access(h);
      auto bitmask_acc = bitmask_buf.get_access(h);
      auto out_acc = out_buf.get_access(h);

      h.single_task([=]() {
        CuckooHashtable<uint32_t, uint32_t,  StaticSimpleHasher<buf_size>, StaticSimpleHasherWithOffset<buf_size, 1>> 
            ht(buf_size, keys_acc.get_pointer(), vals_acc.get_pointer(), 
                    bitmask_acc.get_pointer(), hasher1, hasher2, EMPTY_KEY);
        
        out_acc[0] = ht.insert(5, 5, 0);
        out_acc[1] = ht.insert(15, 15, 0);
        out_acc[2] = ht.insert(20, 20, 0);
        out_acc[3] = ht.insert(0, 0, 0);
      });
    });

  }
  
  ASSERT_EQ(out[0], true);
  ASSERT_EQ(out[1], true);
  ASSERT_EQ(out[2], false);
  ASSERT_EQ(out[3], false);
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
