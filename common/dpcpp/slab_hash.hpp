#pragma once

#include "dpcpp_common.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <math.h>
#include <optional>

namespace SlabHash {
constexpr size_t SUBGROUP_SIZE = 16;
constexpr size_t SLAB_SIZE_MULTIPLIER = 64;
constexpr size_t SLAB_SIZE = SLAB_SIZE_MULTIPLIER * SUBGROUP_SIZE;

constexpr size_t CLUSTER_SIZE = 2048;

constexpr size_t BUCKETS_COUNT = 512;

constexpr size_t EMPTY_UINT32_T = std::numeric_limits<uint32_t>::max();


template <size_t A, size_t B, size_t P>
struct DefaultHasher {
  size_t operator()(const uint32_t &k) {
    return ((A * k + B) % P) % BUCKETS_COUNT;
  };
};

template <typename T>
struct SlabNode {
  SlabNode(T el) {
    for (int i = 0; i < SLAB_SIZE; i++) {
      data[i] = el;
    }
  }

  T data[SLAB_SIZE];
  sycl::device_ptr<SlabNode<T>> next = nullptr;
};

template <typename T>
struct SlabList {
  SlabList() = default;
  SlabList(T empty, sycl::queue &q) {
    auto tmp = sycl::device_ptr<SlabNode<T>>(
        sycl::malloc_device<SlabNode<T>>(1, q));

    q.parallel_for(1, [=](auto &i) {
       *(tmp + i) = SlabNode<T>(empty);
     }).wait();

    root = tmp;
  }

  void clear(sycl::queue &q) { sycl::free(root, q); }

  sycl::device_ptr<SlabNode<T>> root;
};

template <typename T>
struct HeapMaster {
  HeapMaster(sycl::queue &q) : _q(q) {
    _heap = sycl::malloc_device<SlabNode<T>>(CLUSTER_SIZE, q);
    _head = _heap;
  }

  ~HeapMaster() { sycl::free(_heap, _q); }

  sycl::device_ptr<SlabNode<T>> malloc_node() {
    sycl::device_ptr<SlabNode<T>> ret;
    while (sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(&_lock))
               .fetch_or(1)) {
    }
    ret = _head;
    _head++;
    sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(&_lock)).fetch_and(0);
    return ret;
  }

  uint32_t _lock = 0;
  sycl::device_ptr<SlabNode<T>> _heap;
  sycl::device_ptr<SlabNode<T>> _head;
  sycl::queue &_q;
};


template <typename T>
struct AllocAdapter {
  AllocAdapter(size_t bucket_size, T empty, sycl::queue &q) : _q(q), _heap(q) {
    _data.resize(bucket_size);
    _lock.assign(ceil((float)bucket_size / sizeof(uint32_t)), 0);

    for (auto &e : _data) {
      e = SlabList<T>(empty, q);
    }
  }

  ~AllocAdapter() {
    for (auto &e : _data) {
      e.clear(_q);
    }
  }

  std::vector<SlabList<T>> _data;
  std::vector<uint32_t> _lock;
  HeapMaster<T> _heap;
  sycl::queue &_q;
};

template <typename K, typename T, typename Hash>
class SlabHashTable {
public:
  SlabHashTable() = default;
  SlabHashTable(K empty, Hash hasher,
                sycl::global_ptr<SlabList<std::pair<K, T>>> lists,
                sycl::nd_item<1> &it,
                sycl::device_ptr<SlabNode<std::pair<K, T>>> &iter,
                sycl::global_ptr<uint32_t> lock,
                HeapMaster<std::pair<K, T>> &heap)
      : _lists(lists), _gr(it.get_group()), _it(it), _empty(empty),
        _hasher(hasher), _iter(iter), _ind(_it.get_local_id()), _lock(lock),
        _heap(heap){};

  void insert(K key, T val) {
    _key = key;
    _val = val;

    if (_ind == 0) {
      _iter = (_lists + _hasher(key))->root;
    }
    sycl::group_barrier(_gr);

    while (1) {
      while (_iter != nullptr) {
        if (insert_in_node()) {
          return;
        } else if (_ind == 0) {
          _prev = _iter;
          _iter = _iter->next;
        }

        sycl::group_barrier(_gr);
      }
      if (_ind == 0) alloc_node();
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
  void alloc_node() {
    lock();
    if (_prev->next == nullptr) {
      _prev->next = _heap.malloc_node();
      *_prev->next = SlabNode<std::pair<K, T>>({_empty, T()});
    }
    unlock();
    _iter = _prev->next;

  }

  void lock() {
    auto tmp = _hasher(_key);
    int i = 0;
    while (sycl::atomic<uint32_t>((_lock + (tmp / (CHAR_BIT * sizeof(uint32_t)))))
               .fetch_or(1 << (tmp % (CHAR_BIT * sizeof(uint32_t)))) & (1 << (tmp % (CHAR_BIT * sizeof(uint32_t))))) {
    }
  }

  void unlock() {
    auto tmp = _hasher(_key);
    sycl::atomic<uint32_t>((_lock + (tmp / (CHAR_BIT * sizeof(uint32_t)))))
        .fetch_and(~(1 << (tmp %(CHAR_BIT * sizeof(uint32_t)))));
  }

  bool insert_in_node() {
    bool total_found = false;
    bool find = false;

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (SLAB_SIZE_MULTIPLIER - 1);
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
        bool done = _ind == j ? sycl::ext::oneapi::atomic_ref<
                                    K, sycl::ext::oneapi::memory_order::acq_rel,
                                    sycl::ext::oneapi::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    _iter->data[i].first)
                                    .compare_exchange_strong(tmp_empty, _key)
                              : false;
        sycl::group_barrier(_gr);
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

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (SLAB_SIZE_MULTIPLIER - 1);
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
  sycl::global_ptr<uint32_t> _lock;
  sycl::device_ptr<SlabNode<std::pair<K, T>>> &_iter;
  sycl::device_ptr<SlabNode<std::pair<K, T>>> _prev;
  HeapMaster<std::pair<K, T>> &_heap;
  sycl::group<1> _gr;
  sycl::nd_item<1> &_it;
  size_t _ind;

  K _empty;
  Hash _hasher;

  K _key;
  T _val;

  std::optional<T> _ans;
};

} // namespace SlabHash