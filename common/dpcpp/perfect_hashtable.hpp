#pragma once

#include <CL/sycl.hpp>

template <class Key, class T,
          sycl::access::address_space Space =
              sycl::access::address_space::global_space>
class PerfectHashTable {
public:
  PerfectHashTable(size_t hash_size, sycl::multi_ptr<T, Space> vals,
                   Key min_key)
      : _vals(vals), _hash_size(hash_size), _hasher(hash_size, min_key) {}

  bool add(Key key, T val) {
    sycl::atomic<Key, Space>(_vals + _hasher(key)).fetch_add(val);
    return true;
  }

  bool insert(Key key, T val) {
    sycl::atomic<Key, Space>(_vals + _hasher(key)).store(val);
    return true;
  }

  const T at(const Key &key) const {
    return sycl::atomic<Key, Space>(_vals + _hasher(key)).load();
  }

private:
  class PerfectHashFunction {
  public:
    PerfectHashFunction(size_t hash_size, Key min_key)
        : hash_size(hash_size), min_key(min_key) {}

    size_t operator()(Key key) const { return key - min_key; }

  private:
    size_t hash_size;
    Key min_key;
  };

  PerfectHashFunction _hasher;
  size_t _hash_size;

  sycl::multi_ptr<T, Space> _vals;
};