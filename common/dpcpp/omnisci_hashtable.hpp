#pragma once
#include "dpcpp_common.hpp"
#include "hashfunctions.hpp"
#include <iostream>
#include <memory>
#include <numeric>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

namespace OmniSci {
template <typename K, typename V, typename H> class HashTable {
public:
  HashTable(const std::vector<K> &keys_vec, K empty_key, H hasher,
            size_t ht_size, size_t distinct_size, sycl::queue &q)
      : q(q), keys(keys_vec), hash_table(ht_size), count_buffer(ht_size),
        position_buffer(ht_size), id_buffer(keys_vec.size()) {

    ht_fields tmp_f = {empty_key, ht_size,     distinct_size, keys_vec.size(),

                       ht_size,   ht_size * 2, ht_size * 3,

                       hasher};

    f = std::make_unique<ht_fields>(tmp_f);

    // todo parallel init?
    auto ht = hash_table.get_host_access();
    auto cnt = count_buffer.get_host_access();
    auto pos = position_buffer.get_host_access();
    auto id = id_buffer.get_host_access();
    for (int i = 0; i < ht_size; i++) {
      ht[i] = empty_key;
    }
    for (int i = 0; i < ht_size; i++) {
      cnt[i] = 0;
    }
    for (int i = 0; i < ht_size; i++) {
      pos[i] = 0;
    }
    for (int i = 0; i < keys_vec.size(); i++) {
      id[i] = 0;
    }
  }

  void build_table() {
    q.submit([&](sycl::handler &cgh) {
       ht_fields loc_f = *f;
       auto ht = hash_table.get_access(cgh);
       auto ks = keys.get_access(cgh);

       cgh.parallel_for(loc_f.keys_size, [=](auto i) {
         auto h = loc_f.hasher(ks[i]);

         K expected_key = loc_f.empty_key;
         bool success = sycl::atomic<K>(ht.get_pointer() + h)
                            .compare_exchange_strong(expected_key, ks[i]);
         if (success || expected_key == ks[i]) {
           return;
         }
         auto h_probe = (h + 1) % loc_f.ht_size;

         while (h_probe != h) {
           expected_key = loc_f.empty_key;
           success = sycl::atomic<K>(ht.get_pointer() + h_probe)
                         .compare_exchange_strong(expected_key, ks[i]);
           if (success || expected_key == ks[i]) {
             return;
           }
           h_probe = (h_probe + 1) % loc_f.ht_size;
         }
       });
     }).wait();
  }

  void build_id_buffer() {
    build_count_buffer();
    build_pos_buffer();

    q.submit([&](sycl::handler &cgh) {
       ht_fields loc_f = *f;
       auto ht = hash_table.get_access(cgh);
       auto ks = keys.get_access(cgh);
       auto cnt = count_buffer.get_access(cgh);
       auto pos = position_buffer.get_access(cgh);
       auto id = id_buffer.get_access(cgh);

       cgh.parallel_for(loc_f.keys_size, [=](auto i) {
         auto h = loc_f.hasher(ks[i]);

         if (ht[h] == ks[i]) {
           auto id_ind_off =
               sycl::atomic<size_t>(cnt.get_pointer() + h).fetch_add(1);
           auto id_pos = pos[h];
           id[id_pos + id_ind_off] = i;
           return;
         }
         auto h_probe = (h + 1) % loc_f.ht_size;

         while (h_probe != h) {
           if (ht[h_probe] == ks[i]) {
             auto id_ind_off =
                 sycl::atomic<size_t>(cnt.get_pointer() + h_probe).fetch_add(1);
             auto id_pos = pos[h_probe];
             id[id_pos + id_ind_off] = i;
             return;
           }
           h_probe = (h_probe + 1) % loc_f.ht_size;
         }
       });
     }).wait();
  }

  void build_count_buffer() {
    q.submit([&](sycl::handler &cgh) {
       ht_fields loc_f = *f;
       auto ht = hash_table.get_access(cgh);
       auto ks = keys.get_access(cgh);
       auto cnt = count_buffer.get_access(cgh);

       cgh.parallel_for(loc_f.keys_size, [=](auto i) {
         auto h = loc_f.hasher(ks[i]);

         if (ht[h] == ks[i]) {
           sycl::atomic<size_t>(cnt.get_pointer() + h).fetch_add(1);
           return;
         }
         auto h_probe = (h + 1) % loc_f.ht_size;

         while (h_probe != h) {
           if (ht[h_probe] == ks[i]) {
             sycl::atomic<size_t>(cnt.get_pointer() + h_probe).fetch_add(1);
             return;
           }
           h_probe = (h_probe + 1) % loc_f.ht_size;
         }
       });
     }).wait();
  }

  void build_pos_buffer() {
    auto policy = oneapi::dpl::execution::make_device_policy<class mypolicy>(q);
    oneapi::dpl::exclusive_scan(policy, oneapi::dpl::begin(count_buffer),
                                oneapi::dpl::end(count_buffer),
                                oneapi::dpl::begin(position_buffer), 0);

    // todo memset?
    auto cnt = count_buffer.get_host_access();
    for (int i = 0; i < f->ht_size; i++) {
      cnt[i] = 0;
    }
  }

  std::pair<std::vector<size_t>, std::vector<size_t>>
  lookup(const std::vector<K> &other_keys) {
    sycl::buffer<K> other_keys_buf(other_keys);
    sycl::buffer<size_t> join_position_buffer =
        build_join_position_buffer(other_keys.size(), other_keys_buf);
    size_t join_size = 0;

    {
      auto jpos_buf = join_position_buffer.get_host_access();
      join_size = jpos_buf[other_keys.size() - 1];
    }

    if (join_size == 0) {
      return {{}, {}};
    }

    std::vector<size_t> left(join_size);
    std::vector<size_t> right(join_size);

    {
      sycl::buffer<size_t> l_buf(left);
      sycl::buffer<size_t> r_buf(right);

      q.submit([&](sycl::handler &cgh) {
         ht_fields loc_f = *f;
         auto ht = hash_table.get_access(cgh);
         auto ht_cnt = count_buffer.get_access(cgh);
         auto ht_pos = position_buffer.get_access(cgh);
         auto ht_ids = id_buffer.get_access(cgh);

         auto ks = other_keys_buf.get_access(cgh);
         auto jpos_buf = join_position_buffer.get_access(cgh);

         auto l = l_buf.get_access(cgh);
         auto r = r_buf.get_access(cgh);

         cgh.parallel_for(loc_f.keys_size, [=](auto i) {
           auto id = i == 0 ? 0 : jpos_buf[i - 1];

           auto h = loc_f.hasher(ks[i]);
           auto ht_id = h;

           if (ht[h] != ks[i]) {
             auto h_probe = (h + 1) % loc_f.ht_size;

             while (h_probe != h) {
               if (ht[h_probe] == ks[i]) {
                 ht_id = h_probe;
                 break;
               }
               h_probe = (h_probe + 1) % loc_f.ht_size;
             }
             if (h_probe == h)
               return;
           }
           auto join_entries = ht_cnt[ht_id];

           if (join_entries == 0)
             return; // impossible?

           for (int j = 0; j < join_entries; j++) {
             l[id + j] = i;
             r[id + j] = ht_ids[ht_pos[ht_id] + j];
           }
         });
       }).wait();
    }

    return {left, right};
  }

  sycl::buffer<size_t>
  build_join_position_buffer(size_t buffer_size,
                             sycl::buffer<K> &other_keys_buf) {
    sycl::buffer<size_t> join_count_buffer(buffer_size);

    { // parallel??
      auto jcb = join_count_buffer.get_host_access();
      for (int i = 0; i < buffer_size; i++) {
        jcb[i] = 0;
      }
    }

    q.submit([&](sycl::handler &cgh) {
       ht_fields loc_f = *f;
       auto ht = hash_table.get_access(cgh);
       auto ks = other_keys_buf.get_access(cgh);
       auto cnt = join_count_buffer.get_access(cgh);
       auto ht_cnt = count_buffer.get_access(cgh);

       cgh.parallel_for(loc_f.keys_size, [=](auto i) {
         auto h = loc_f.hasher(ks[i]);

         if (ht[h] == ks[i]) {
           sycl::atomic<size_t>(cnt.get_pointer() + i).fetch_add(ht_cnt[h]);
           return;
         }
         auto h_probe = (h + 1) % loc_f.ht_size;

         // or empty
         while (h_probe != h) {
           if (ht[h_probe] == ks[i]) {
             sycl::atomic<size_t>(cnt.get_pointer() + i)
                 .fetch_add(ht_cnt[h_probe]);
             return;
           }
           h_probe = (h_probe + 1) % loc_f.ht_size;
         }
       });
     }).wait();

    auto policy = oneapi::dpl::execution::make_device_policy<class mysecondpolicy>(q);
    oneapi::dpl::inclusive_scan(policy, oneapi::dpl::begin(join_count_buffer),
                                oneapi::dpl::end(join_count_buffer),
                                oneapi::dpl::begin(join_count_buffer));
    return join_count_buffer;
  }

  void dump_buffer() {
    auto ht = hash_table.get_host_access();
    auto cnt = count_buffer.get_host_access();
    auto pos = position_buffer.get_host_access();
    auto id = id_buffer.get_host_access();

    for (int i = 0; i < f->ht_size; i++) {
      std::cout << ht[i] << ' ';
    }
    std::cout << " | ";
    for (int i = 0; i < f->ht_size; i++) {
      std::cout << pos[i] << ' ';
    }
    std::cout << " | ";
    for (int i = 0; i < f->ht_size; i++) {
      std::cout << cnt[i] << ' ';
    }
    std::cout << " | ";
    for (int i = 0; i < f->keys_size; i++) {
      std::cout << id[i] << ' ';
    }
    std::cout << std::endl;
  }

  size_t get_ht_size() { return f->ht_size; }

  size_t get_buffer_size() { return f->ht_size * 3 + f->keys_size; }

private:
  struct ht_fields {
    K empty_key;
    size_t ht_size;
    size_t distinct_size;
    size_t keys_size;

    size_t pos_offset;
    size_t count_offset;
    size_t id_offset;

    H hasher;
  };
  sycl::buffer<K> keys;
  sycl::buffer<K> hash_table;
  sycl::buffer<size_t> position_buffer;
  sycl::buffer<size_t> count_buffer;
  sycl::buffer<size_t> id_buffer;
  sycl::queue &q;

  std::unique_ptr<ht_fields> f;
};
} // namespace OmniSci
