#pragma once
#include "common/common.hpp"

namespace join_helpers {

template <class Key, class Val1, class Val2>
using ColJoinedTableTy =
    std::pair<std::vector<Key>,
              std::pair<std::vector<Val1>, std::vector<Val2>>>;

template <class Key, class V1, class V2>
using RowJoinedTableTy = std::vector<std::pair<Key, std::pair<V1, V2>>>;

template <class K, class V1, class V2>
bool is_malformed(const ColJoinedTableTy<K, V1, V2> &t) {
  return t.first.size() != t.second.first.size() ||
         t.first.size() != t.second.second.size();
}

template <class K, class V1, class V2>
size_t get_size(const ColJoinedTableTy<K, V1, V2> &t) {
  if (is_malformed(t))
    throw std::invalid_argument("ColJoinedTableTy is malformed.");
  return t.first.size();
}

template <class K, class V1, class V2>
size_t get_size(const RowJoinedTableTy<K, V1, V2> &t) {
  return t.size();
}

template <class Key, class V1, class V2>
RowJoinedTableTy<Key, V1, V2>
to_row_store(const ColJoinedTableTy<Key, V1, V2> &t) {
  RowJoinedTableTy<Key, V1, V2> res;
  if (is_malformed(t))
    throw std::invalid_argument("ColJoinedTableTy is malformed.");

  for (unsigned i = 0; i < t.first.size(); ++i) {
    res.push_back({t.first[i], {t.second.first[i], t.second.second[i]}});
  }

  return res;
}

template <class Key, class V1, class V2>
ColJoinedTableTy<Key, V1, V2>
to_col_store(const RowJoinedTableTy<Key, V1, V2> &t) {
  ColJoinedTableTy<Key, V1, V2> res;
  for (auto &it : t) {
    res.first.push_back(it.first);
    res.second.first.push_back(it.second.first);
    res.second.second.push_back(it.second.second);
  }
  return res;
}

template <class Key, class Val1, class Val2>
std::ostream &operator<<(std::ostream &os,
                         const ColJoinedTableTy<Key, Val1, Val2> &t) {
  if (is_malformed(t))
    throw std::invalid_argument("ColJoinedTableTy is malformed.");

  bool first = true;
  auto sz = t.first.size();
  for (unsigned i = 0; i < sz; ++i) {
    if (!first)
      os << "\n";
    os << t.first[i] << " " << t.second.first[i] << " " << t.second.second[i];
    first = false;
  }
  return os;
}

template <class K, class V1, class V2>
ColJoinedTableTy<K, V1, V2>
seq_join(const std::vector<K> &a_keys, const std::vector<V1> &a_vals,
         const std::vector<K> &b_keys, const std::vector<V2> &b_vals) {
  ColJoinedTableTy<K, V1, V2> result;
  std::vector<K> keys;
  std::vector<V1> vals1;
  std::vector<V2> vals2;

  for (size_t i = 0; i < a_keys.size(); ++i) {
    for (size_t j = 0; j < b_keys.size(); ++j) {
      if (a_keys[i] == b_keys[j]) {
        keys.push_back(a_keys[i]);
        vals1.push_back(a_vals[i]);
        vals2.push_back(b_vals[j]);
      }
    }
  }
  return {keys, {vals1, vals2}};
}

template <class K, class V1, class V2>
bool operator==(const ColJoinedTableTy<K, V1, V2> &t1,
                const ColJoinedTableTy<K, V1, V2> &t2) {
  if (is_malformed(t1) || is_malformed(t2))
    throw std::invalid_argument("ColJoinedTableTy is malformed.");

  return to_row_store(t1) == to_row_store(t2);
}

template <class K, class V1, class V2>
bool operator==(const RowJoinedTableTy<K, V1, V2> &t1,
                const RowJoinedTableTy<K, V1, V2> &t2) {
  auto temp1 = t1;
  auto temp2 = t2;

  std::sort(temp1);
  std::sort(temp2);

  return temp1 == temp2;
}

template <class K, class V1, class V2>
ColJoinedTableTy<K, V1, V2> zip(const std::vector<K> &keys,
                                const std::vector<V1> &v1,
                                const std::vector<V2> &v2) {
  ColJoinedTableTy<K, V1, V2> res;

  res.first = keys;
  res.second.first = v1;
  res.second.second = v2;

  return res;
}

} // namespace join_helpers