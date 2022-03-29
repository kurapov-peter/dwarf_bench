#pragma once
#include "dpcpp_common.hpp"
#include "hashfunctions.hpp"
#include <iostream>
#include <memory>
#include <numeric>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

struct JoinOneToMany {
  sycl::global_ptr<size_t> vals;
  size_t size;
};

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

  std::vector<JoinOneToMany> lookup(const std::vector<K> &other_keys) {
    std::vector<JoinOneToMany> answer(other_keys.size());

    {
      sycl::buffer<JoinOneToMany> join(answer);
      sycl::buffer<K> keys(other_keys);

      q.submit([&](sycl::handler &cgh) {
         ht_fields loc_f = *f;
         auto ht = hash_table.get_access(cgh);
         auto ht_cnt = count_buffer.get_access(cgh);
         auto ht_pos = position_buffer.get_access(cgh);
         auto ht_ids = id_buffer.get_access(cgh);

         auto ks = keys.get_access(cgh);
         auto j = join.get_access(cgh);

         cgh.parallel_for(other_keys.size(), [=](auto i) {
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
           j[i].vals = (ht_ids.get_pointer() + ht_pos[ht_id]);
           j[i].size = ht_cnt[ht_id];
         });
       }).wait();
    }

    return answer;
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
