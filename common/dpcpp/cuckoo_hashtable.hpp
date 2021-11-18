#include <CL/sycl.hpp>
#include <algorithm>

#include "dpcpp_common.hpp"
#include "hashfunctions.hpp"

template <class Key, class Val, class Hasher1, class Hasher2>
class CuckooHashtable {
private:
  sycl::global_ptr<Key> _keys;
  sycl::global_ptr<Val> _vals;

  const size_t _input_size;
  Hasher1 _hasher1;
  Hasher2 _hasher2;
  const Key _EMPTY_KEY = std::numeric_limits<Key>::max();
  sycl::global_ptr<uint32_t> _bitmask;
  static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);
  const size_t max_iter = 1e5;

public:
  explicit CuckooHashtable(const size_t input_size, sycl::global_ptr<Key> keys,
                           sycl::global_ptr<Val> vals,
                           sycl::global_ptr<uint32_t> bitmask, Hasher1 hasher1,
                           Hasher2 hasher2)
      : _input_size(input_size), _keys(keys), _vals(vals), _bitmask(bitmask),
        _hasher1(hasher1), _hasher2(hasher2) {}

  const std::pair<Val, bool> at(Key key) const {
    auto pos1 = _hasher1(key);
    auto pos2 = _hasher2(key);
    if (_keys[pos1] == key)
      return {_vals[pos1], true};
    if (_keys[pos2] == key)
      return {_vals[pos2], true};
    return {{}, false};
  }

  bool has(Key key) const {
    return _keys[_hasher1(key)] == key || _keys[_hasher2(key)] == key;
  }

  bool insert(Key key, Val value) {
    size_t pos = _hasher1(key);
    for (int cnt = 0; cnt < std::min(_input_size, max_iter); cnt++) {
      lock(pos);
      if (_keys[pos] == _EMPTY_KEY) {
        _keys[pos] = key;
        _vals[pos] = value;

        unlock(pos);
        return true;
      }
      std::swap(key, _keys[pos]);
      std::swap(value, _vals[pos]);
      unlock(pos);
      if (pos == _hasher1(key))
        pos = _hasher2(key);
      else
        pos = _hasher1(key);
    }
    return false;
  }

  void lock(size_t pos) {
    uint32_t present;
    uint32_t major_idx = pos / elem_sz;
    uint8_t minor_idx = pos % elem_sz;
    uint32_t mask = uint32_t(1) << minor_idx;
    do {
      present = sycl::atomic<uint32_t>(_bitmask + major_idx).fetch_or(mask);
    } while (present & mask);
  }

  void unlock(size_t pos) {
    uint32_t major_idx = pos / elem_sz;
    uint8_t minor_idx = pos % elem_sz;
    uint32_t mask = uint32_t(1) << minor_idx;
    sycl::atomic<uint32_t>(_bitmask + major_idx).fetch_and(~mask);
  }
};