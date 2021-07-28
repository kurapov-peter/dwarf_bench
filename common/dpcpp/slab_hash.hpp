#pragma once

#include "dpcpp_common.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <optional>

constexpr size_t SUBGROUP_SIZE = 16;
constexpr size_t CONST = 64;
constexpr size_t SLAB_SIZE = CONST * SUBGROUP_SIZE;

constexpr size_t CLUSTER_SIZE = 1024;

constexpr size_t BUCKETS_COUNT = 128;

constexpr size_t EMPTY_UINT32_T = std::numeric_limits<uint32_t>::max();

namespace SlabHashHashers {
template <size_t A, size_t B, size_t P> struct DefaultHasher {
  size_t operator()(const uint32_t &k) {
    return ((A * k + B) % P) % BUCKETS_COUNT;
  };
};
} // namespace SlabHashHashers

template <typename T> struct SlabNode {
  SlabNode(T el, size_t size, sycl::queue &q) : _q(q) {
    data = sycl::malloc_shared<T>(size, q);
    for (int i = 0; i < size; i++) {
      data[i] = el;
    }
  }

  void clear() { sycl::free(data, _q); }

  T *data;
  sycl::queue &_q;
  sycl::global_ptr<SlabNode<T>> next = nullptr;
};

template <typename T> struct SlabList {
  SlabList() = default;
  SlabList(T empty, size_t slab_size, sycl::queue &q) : _q(q) {
      for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
       root = sycl::global_ptr<SlabNode<T>>(
          sycl::malloc_shared<SlabNode<T>>(CLUSTER_SIZE, q));
      *(root + i) = SlabNode<T>(empty, slab_size, q);
      (root + i)->next = (root + i + 1);
    }
  }

  void clear() {
    for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
      (root + i)->clear();
      (root + i)->next = (root + i + 1);
    }
    sycl::free(root, _q);
  }

  sycl::global_ptr<SlabNode<T>> root;
  sycl::queue &_q;
};

namespace SlabHashHelpers {
template <typename T> struct AllocAdapter {
  AllocAdapter(size_t bucket_size, size_t slab_size, T empty, sycl::queue &q) {
    _data.resize(bucket_size);
    for (auto &e : _data) {
      e = SlabList<T>(empty, slab_size, q);
    }
  }

  ~AllocAdapter() {
    for (auto &e : _data) {
      e.clear();
    }
  }

  std::vector<SlabList<T>> _data;
};
} // namespace SlabHashHelpers

template <typename K, typename T, typename Hash> class SlabHash {
public:
  SlabHash() = default;
  SlabHash(K empty, Hash hasher,
           sycl::global_ptr<SlabList<std::pair<K, T>>> lists,
           sycl::nd_item<1> &it,
           sycl::global_ptr<SlabNode<std::pair<K, T>>> &iter)
      : _lists(lists), _gr(it.get_group()), _it(it), _empty(empty),
        _hasher(hasher), _iter(iter), _ind(_it.get_local_id()){};

  void insert(K key, T val) {
    _key = key;
    _val = val;

    if (_ind == 0) {
      _iter = (_lists + _hasher(key))->root;
    }
    sycl::group_barrier(_gr);

    while (_iter != nullptr) {
      if (insert_in_node()) {
        break;
      } else if (_ind == 0) {
        _iter = _iter->next;
      }

      sycl::group_barrier(_gr);
    }
  }

  std::optional<T> find(K key) {
    _key = key;
    _ans = std::nullopt;

    if (_ind == 0) {
      _iter = (_lists + _hasher(key))->root;
    }
    sycl::group_barrier(_gr);

    while (_iter != nullptr) {
      if (find_in_node()) {
        break;
      } else if (_ind == 0) {
        _iter = _iter->next;
      }

      sycl::group_barrier(_gr);
    }

    return _ans;
  }

private:
  bool insert_in_node() {
    bool total_found = false;
    bool find = false;

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (CONST - 1);
         i += SUBGROUP_SIZE) {
      find = ((_iter->data[i].first) == _empty);
      sycl::group_barrier(_gr);
      total_found = sycl::any_of_group(_gr, find);

      if (total_found) {
        if (insert_in_subgroup(find, i)) {
          return true;
        }
      }
    }

    return false;
  }

  bool insert_in_subgroup(bool find, int i) {
    for (int j = 0; j < SUBGROUP_SIZE; j++) {
      if (cl::sycl::group_broadcast(_gr, find, j)) {
        K tmp_empty = _empty;
        bool done = _ind == j ? sycl::ONEAPI::atomic_ref<
                                    K, sycl::ONEAPI::memory_order::acq_rel,
                                    sycl::ONEAPI::memory_scope::system,
                                    sycl::access::address_space::global_space>(
                                    _iter->data[i].first)
                                    .compare_exchange_strong(tmp_empty, _key)
                              : false;
        if (done) {
          _iter->data[i].second = _val;
        }
        if (cl::sycl::group_broadcast(_gr, done, j)) {
          return true;
        }
      }
    }

    return false;
  }

  bool find_in_node() {
    bool find = false;
    bool total_found = false;

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (CONST - 1);
         i += SUBGROUP_SIZE) {
      find = ((_iter->data[i].first) == _key);
      sycl::group_barrier(_gr);
      total_found = sycl::any_of_group(_gr, find);

      if (total_found) {
        find_in_subgroup(find, i);
        return true;
      }
    }

    return false;
  }

  void find_in_subgroup(bool find, int i) {
    for (int j = 0; j < SUBGROUP_SIZE; j++) {
      if (cl::sycl::group_broadcast(_gr, find, j)) {
        T tmp;
        if (_ind == j)
          tmp = _iter->data[i].second; // todo index shuffle

        _ans = std::optional<T>{cl::sycl::group_broadcast(_gr, tmp, j)};
        break;
      }
    }
  }

  sycl::global_ptr<SlabList<std::pair<K, T>>> _lists;
  sycl::global_ptr<SlabNode<std::pair<K, T>>> &_iter;
  sycl::group<1> _gr;
  sycl::nd_item<1> &_it;
  size_t _ind;

  K _empty;
  Hash _hasher;

  K _key;
  T _val;

  std::optional<T> _ans;
};