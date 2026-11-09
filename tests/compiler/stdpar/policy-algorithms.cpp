// RUN: %acpp %s --acpp-targets=generic -O0 -std=c++17 -fsyntax-only \
// RUN:   -isystem %S/../../../include/hipSYCL/std/stdpar -I%S/../../../include \
// RUN:   -Wno-deprecated-declarations -Wno-format-security

#include <hipSYCL/common/filesystem.hpp>
#include <hipSYCL/std/stdpar/detail/sycl_glue.hpp>
#include <execution>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>

// Instantiate the supported overloads, including their host-fallback bodies
// (except the pre-existing default-sort issue noted below).
// libc++ policies cannot be copied or default-constructed, and its native
// PSTL lacks several of these overloads (notably copy_if and scans).
template<class Policy>
void instantiate_algorithms(const Policy& policy, int* first, int* last,
                            int* second, int* output) {
  auto unary = [](int x) { return x + 1; };
  auto predicate = [](int x) { return x > 0; };
  auto update = [](int& x) { ++x; };
  auto generate = [] { return 1; };
  std::plus<int> plus;
  std::multiplies<int> multiply;
  std::less<int> less;
  std::equal_to<int> equal;

  std::for_each(policy, first, last, update);
  (void)std::for_each_n(policy, first, 4, update);
  (void)std::transform(policy, first, last, output, unary);
  (void)std::transform(policy, first, last, second, output, plus);
  (void)std::copy(policy, first, last, output);
  (void)std::copy_if(policy, first, last, output, predicate);
  (void)std::copy_n(policy, first, 4, output);
  (void)std::move(policy, first, last, output);
  std::fill(policy, first, last, 1);
  (void)std::fill_n(policy, first, 4, 1);
  std::generate(policy, first, last, generate);
  (void)std::generate_n(policy, first, 4, generate);
  (void)std::remove_copy(policy, first, last, output, 1);
  (void)std::remove_copy_if(policy, first, last, output, predicate);
  (void)std::remove(policy, first, last, 1);
  (void)std::remove_if(policy, first, last, predicate);
  std::replace(policy, first, last, 1, 2);
  std::replace_if(policy, first, last, predicate, 2);
  (void)std::replace_copy(policy, first, last, output, 1, 2);
  (void)std::replace_copy_if(policy, first, last, output, predicate, 2);
  std::reverse(policy, first, last);
  (void)std::reverse_copy(policy, first, last, output);
  (void)std::find(policy, first, last, 1);
  (void)std::find_if(policy, first, last, predicate);
  (void)std::find_if_not(policy, first, last, predicate);
  (void)std::find_end(policy, first, last, second, second + 2);
  (void)std::find_end(policy, first, last, second, second + 2, equal);
  (void)std::find_first_of(policy, first, last, second, second + 2);
  (void)std::find_first_of(policy, first, last, second, second + 2, equal);
  (void)std::all_of(policy, first, last, predicate);
  (void)std::any_of(policy, first, last, predicate);
  (void)std::none_of(policy, first, last, predicate);
  (void)std::count(policy, first, last, 1);
  (void)std::count_if(policy, first, last, predicate);
  (void)std::mismatch(policy, first, last, second);
  (void)std::mismatch(policy, first, last, second, equal);
  (void)std::mismatch(policy, first, last, second, second + 4);
  (void)std::mismatch(policy, first, last, second, second + 4, equal);
  (void)std::equal(policy, first, last, second);
  (void)std::equal(policy, first, last, second, equal);
  (void)std::equal(policy, first, last, second, second + 4);
  (void)std::equal(policy, first, last, second, second + 4, equal);
  // The existing device sort cannot deduce Compare without an explicit
  // comparator. Check this signature only; host-fallback.cpp exercises its
  // serial fallback. The explicit-comparator overload is instantiated below.
  static_assert(std::is_same_v<decltype(std::sort(policy, first, last)), void>);
  std::sort(policy, first, last, less);
  (void)std::is_sorted(policy, first, last);
  (void)std::is_sorted(policy, first, last, less);
  (void)std::is_sorted_until(policy, first, last);
  (void)std::is_sorted_until(policy, first, last, less);
  (void)std::merge(policy, first, last, second, second + 4, output);
  (void)std::merge(policy, first, last, second, second + 4, output, less);
  (void)std::min_element(policy, first, last);
  (void)std::min_element(policy, first, last, less);
  (void)std::max_element(policy, first, last);
  (void)std::max_element(policy, first, last, less);

  (void)std::transform_reduce(policy, first, last, second, 0);
  (void)std::transform_reduce(policy, first, last, second, 0, plus, multiply);
  (void)std::transform_reduce(policy, first, last, 0, plus, unary);
  (void)std::reduce(policy, first, last);
  (void)std::reduce(policy, first, last, 0);
  (void)std::reduce(policy, first, last, 0, plus);
  (void)std::inclusive_scan(policy, first, last, output);
  (void)std::inclusive_scan(policy, first, last, output, plus);
  (void)std::inclusive_scan(policy, first, last, output, plus, 0);
  (void)std::exclusive_scan(policy, first, last, output, 0);
  (void)std::exclusive_scan(policy, first, last, output, 0, plus);
  (void)std::transform_inclusive_scan(policy, first, last, output, plus, unary);
  (void)std::transform_inclusive_scan(policy, first, last, output, plus, unary, 0);
  (void)std::transform_exclusive_scan(policy, first, last, output, 0, plus, unary);
}

void check_algorithms(int* first, int* last, int* second, int* output) {
  instantiate_algorithms(std::execution::par_unseq, first, last, second, output);
#ifndef _LIBCPP_VERSION
  instantiate_algorithms(std::execution::par, first, last, second, output);
#endif
}
