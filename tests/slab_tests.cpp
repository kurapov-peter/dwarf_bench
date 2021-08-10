#include "common/dpcpp/slab_hash.hpp"
#include <gtest/gtest.h>
#include <math.h>
#include <vector>

using std::pair;
using namespace SlabHash;

TEST(SlabHash, insert) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {{1, 2}, {5, 2}, {101, 3},
                                                    {5, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q, 3);

  std::vector<uint8_t> checks(6, 0);

  {
    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
          adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<uint8_t> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);
        
        cgh.parallel_for<class insert_test_slab>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
 
         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           ht.insert(tests[i].first, tests[i].second);
         }
       });
     }).wait();

     q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto checks = sycl::accessor(checks_buf, cgh, sycl::read_write);
        auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

        cgh.parallel_for<class insert_test_slab_check>(6, [=](auto &it) {
         size_t ind = it;
 
          DefaultHasher<13,24,343> h;
         auto list = (*(adap_acc.get_pointer()))._data[h(tests[ind].first)];
         for (int i = 0; i < SLAB_SIZE; i++) {
           if (list.root->data[i] == tests[ind]) {
             checks[ind] = 1;
             break;
           }
         }
       });
     }).wait();
  }

  
  for (auto e: checks) {
    EXPECT_EQ(e, 1);
  }
}
/*
TEST(SlabHash, find) {
  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  std::vector<SlabList<pair<uint32_t, uint32_t>>> lists(BUCKETS_COUNT);
  for (auto &e : lists) {
    e.root = sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>(
        sycl::malloc_shared<SlabNode<pair<uint32_t, uint32_t>>>(CLUSTER_SIZE,
                                                                q));

    *(e.root) = SlabNode<pair<uint32_t, uint32_t>>({EMPTY_UINT32_T, 0});
  }

  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {21312, 5}, {3, 0}, {10, 10}};

  DefaultHasher<13, 24, 343> h;

  for (auto &e : testUniv) {
    auto r = lists[h(e.first)].root;

    for (int i = 0; i < SLAB_SIZE; i++) {
      if (r->data[i].first == EMPTY_UINT32_T) {
        r->data[i] = e;
        break;
      }
    }
  }

  HeapMaster<pair<uint32_t, uint32_t>> heap(q);
  std::vector<uint32_t> lock(6);

  std::vector<pair<bool, bool>> checks(6);
  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
    sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>>
          heap_buf(&heap, sycl::range<1> {1});
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);
    sycl::buffer<uint32_t> lock_buf(lock);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);
       auto heap_acc = sycl::accessor(heap_buf, cgh, sycl::read_write);
       auto lock_acc = sycl::accessor(lock_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
         DefaultHasher<13, 24, 343> h;
         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             EMPTY_UINT32_T, h, l.get_pointer(), it,
             itrs[it.get_group().get_id()], lock_acc.get_pointer(), *heap_acc.get_pointer());

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     }).wait();
  }

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(checks[i].first && checks[i].second);
  }

  for (auto &e : lists) {
    sycl::free(e.root, q);
  }
}

TEST(SlabHash, find_and_insert_together) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {10932, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  AllocAdapter<std::pair<uint32_t, uint32_t>> adap(BUCKETS_COUNT,
                                                   {EMPTY_UINT32_T, 0}, q);

  {
    sycl::buffer<uint32_t> lock_buf(adap._lock);
    // std::cout << "LOCK CREATED\n";
    sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>> heap_buf(
        &adap._heap, sycl::range<1>{1});
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);

       auto heap_acc = sycl::accessor(heap_buf, cgh, sycl::read_write);
       auto lock_acc = sycl::accessor(lock_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHash::SlabHashTable<uint32_t, uint32_t,
                                          SlabHash::DefaultHasher<13, 24, 343>>
                 ht(SlabHash::EMPTY_UINT32_T, h, l.get_pointer(), it,
                    itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                    *heap_acc.get_pointer());

             for (int i = ind * 2; i < ind * 2 + 2; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     }).wait();
  }

  std::vector<pair<bool, bool>> checks(6);
  {
    sycl::buffer<uint32_t> lock_buf(adap._lock);
    // std::cout << "LOCK CREATED\n";
    sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>> heap_buf(
        &adap._heap, sycl::range<1>{1});
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);

       auto heap_acc = sycl::accessor(heap_buf, cgh, sycl::read_write);
       auto lock_acc = sycl::accessor(lock_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab_both>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
         DefaultHasher<13, 24, 343> h;
         SlabHash::SlabHashTable<uint32_t, uint32_t,
                                      SlabHash::DefaultHasher<13, 24, 343>>
             ht(SlabHash::EMPTY_UINT32_T, h, l.get_pointer(), it,
                itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                *heap_acc.get_pointer());

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     }).wait();
  }

  for (auto &e : checks) {
    EXPECT_TRUE(e.first && e.second);
  }
}

TEST(SlabHash, find_and_insert_together_big) {
  std::vector<pair<uint32_t, uint32_t>> testUniv;
  auto f = [&](int i) { return i * i; };

  for (int i = 0; i < 1000; i++) {
    testUniv.push_back({f(i), f(i)});
  }

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 25, SUBGROUP_SIZE};

  AllocAdapter<std::pair<uint32_t, uint32_t>> adap(BUCKETS_COUNT,
                                                   {EMPTY_UINT32_T, 0}, q);

  {
    sycl::buffer<uint32_t> lock_buf(adap._lock);
    // std::cout << "LOCK CREATED\n";
    sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>> heap_buf(
        &adap._heap, sycl::range<1>{1});
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(25);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto heap_acc = sycl::accessor(heap_buf, cgh, sycl::read_write);
       // std::cout << "HEAP ACCESSED\n";
       auto lock_acc = sycl::accessor(lock_buf, cgh, sycl::read_write);
       // std::cout << "LOCK ACCESSED\n";
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHash::SlabHashTable<uint32_t, uint32_t,
                                          SlabHash::DefaultHasher<13, 24, 343>>
                 ht(SlabHash::EMPTY_UINT32_T, h, l.get_pointer(), it,
                    itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                    *heap_acc.get_pointer());

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     }).wait();
  }

  std::vector<pair<bool, bool>> checks(1000);
  {
    sycl::buffer<uint32_t> lock_buf(adap._lock);
    // std::cout << "LOCK CREATED\n";
    sycl::buffer<SlabHash::HeapMaster<pair<uint32_t, uint32_t>>> heap_buf(
        &adap._heap, sycl::range<1>{1});
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(25);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);

    q.submit([&](sycl::handler &cgh) {
       auto heap_acc = sycl::accessor(heap_buf, cgh, sycl::read_write);
       // std::cout << "HEAP ACCESSED\n";
       auto lock_acc = sycl::accessor(lock_buf, cgh, sycl::read_write);
       // std::cout << "LOCK ACCESSED\n";
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);

       cgh.parallel_for<class find_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHash::SlabHashTable<uint32_t, uint32_t,
                                          SlabHash::DefaultHasher<13, 24, 343>>
                 ht(SlabHash::EMPTY_UINT32_T, h, l.get_pointer(), it,
                    itrs[it.get_group().get_id()], lock_acc.get_pointer(),
                    *heap_acc.get_pointer());

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               auto ans = ht.find(tests[i].first);

               if (it.get_local_id() == 0)
                 accChecks[i] = {static_cast<bool>(ans),
                                 ans.value_or(-1) == tests[i].second};
             }
           });
     }).wait();
  }

  for (auto &e : checks) {
    EXPECT_TRUE(e.first && e.second);
  }
}*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
