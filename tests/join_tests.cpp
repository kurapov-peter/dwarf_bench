#include "join/slab_join.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(Join, HelpersSeqJoin) {
  using namespace std;

  vector<int> keys_a = {1, 2, 3, 4, 5, 5, 7};
  vector<int> vals_a = {5, 1, 4, 6, 6, 5, 0};
  vector<int> keys_b = {6, 2, 3, 4, 5, 5, 7};
  vector<int> vals_b = {3, 2, 1, 1, 3, 8, 8};

  auto res = join_helpers::seq_join(keys_a, vals_a, keys_b, vals_b);

  ASSERT_EQ(res.first.size(), res.second.first.size());
  ASSERT_EQ(res.first.size(), res.second.second.size());
  ASSERT_EQ(res.first.size(), 8);

  using namespace join_helpers;
  cout << res << "\n";
}

TEST(Join, HelpersEqual) {
  using namespace std;
  vector<int> keys_a = {1, 2, 3, 4, 5, 5, 7};
  vector<int> vals_a = {5, 1, 4, 6, 6, 5, 0};
  vector<int> keys_b = {6, 2, 3, 4, 5, 5, 7};
  vector<int> vals_b = {3, 2, 1, 1, 3, 8, 8};

  using namespace join_helpers;
  auto a = seq_join(keys_a, vals_a, keys_b, vals_b);
  auto b = seq_join(keys_a, vals_a, keys_b, vals_b);

  ASSERT_EQ(a, b);

  auto a_rows = to_row_store(a);
  auto b_rows = to_row_store(b);

  ASSERT_EQ(a_rows, b_rows);
}

TEST(Join, HelpersConvert) {
  using namespace std;

  vector<int> keys_a = {1, 2, 3, 4, 5, 5, 7};
  vector<int> vals_a = {5, 1, 4, 6, 6, 5, 0};
  vector<int> keys_b = {6, 2, 3, 4, 5, 5, 7};
  vector<int> vals_b = {3, 2, 1, 1, 3, 8, 8};

  using namespace join_helpers;
  auto res = seq_join(keys_a, vals_a, keys_b, vals_b);

  auto rows = to_row_store(res);
  auto converted = to_col_store(rows);

  ASSERT_EQ(res, converted);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}