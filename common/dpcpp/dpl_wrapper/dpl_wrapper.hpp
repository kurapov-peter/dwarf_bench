#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

#include <CL/sycl.hpp>

#ifdef CUDA_COMPILATION
#define SUFFIX(name) name##Cuda
#else
#define SUFFIX(name) name
#endif

namespace DPLWrapper {

template <class Kernel, typename T>
void exclusive_scan(sycl::queue &q, sycl::buffer<T> &src, sycl::buffer<T> &dst,
                    T init_value) {
  auto policy = oneapi::dpl::execution::make_device_policy<Kernel>(q);
  oneapi::dpl::exclusive_scan(policy, oneapi::dpl::begin(src),
                              oneapi::dpl::end(src), oneapi::dpl::begin(dst),
                              init_value);
}

template <class Kernel, typename T, typename F>
void copy_if(cl::sycl::device_selector &sel, sycl::buffer<T> &src,
             sycl::buffer<T> &dst, F func) {
  auto policy = oneapi::dpl::execution::device_policy<Kernel>{sel};
  std::copy_if(policy, oneapi::dpl::begin(src), oneapi::dpl::end(src),
               oneapi::dpl::begin(dst), func);
}

template <class Kernel, typename T>
void sort(cl::sycl::device_selector &sel, sycl::buffer<T> &src) {
  auto policy = oneapi::dpl::execution::device_policy<Kernel>{sel};
  std::sort(policy, oneapi::dpl::begin(src), oneapi::dpl::end(src));
}

} // namespace DPLWrapper