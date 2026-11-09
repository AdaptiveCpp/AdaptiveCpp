// RUN: %acpp %s --acpp-targets=generic -O0 -std=c++17 -fsyntax-only \
// RUN:   -I%S/../../../include/hipSYCL/std/stdpar -I%S/../../../include \
// RUN:   -Wno-deprecated-declarations -Wno-format-security \
// RUN:   -Xclang -verify -Xclang -verify-ignore-unexpected=note
// RUN: %acpp %s --acpp-targets=generic -O0 -std=c++17 -fsyntax-only \
// RUN:   -I%S/../../../include/hipSYCL/std/stdpar -I%S/../../../include \
// RUN:   -DSTDPAR_NUMERIC_FIRST -Wno-deprecated-declarations -Wno-format-security \
// RUN:   -Xclang -verify -Xclang -verify-ignore-unexpected=note

// Interpose the headers explicitly instead of using --acpp-stdpar: its forced
// sycl_glue.hpp include would choose the first header before this test can.
#include <hipSYCL/std/stdpar/detail/stdpar_defs.hpp>

// Mark the real entrypoints, without adding overloads that could repair missing
// forward declarations. diagnose_if does not participate in overload resolution.
// Unevaluated calls below check dispatch without instantiating kernels/fallbacks.
// Allow one or two diagnostics: some entrypoints have both a marked forward
// declaration and a marked definition; others are first declared in <execution>.
#pragma push_macro("HIPSYCL_STDPAR_ENTRYPOINT")
#undef HIPSYCL_STDPAR_ENTRYPOINT
#define HIPSYCL_STDPAR_ENTRYPOINT                                              \
  HIPSYCL_STDPAR_NOINLINE                                                      \
  __attribute__((annotate("hipsycl_stdpar_entrypoint"),                          \
                 diagnose_if(true, "AdaptiveCpp stdpar overload selected",     \
                             "warning")))


#ifdef STDPAR_NUMERIC_FIRST
#include <numeric>
#include <algorithm>
#else
#include <algorithm>
#include <numeric>
#endif

// Qualified lookup is fixed here, before the explicit <execution> include.
// A later declaration must not conceal a missing forward declaration.
template<class Policy>
auto early_fill(Policy&& policy, int* first, int* last)
    -> decltype(std::fill(static_cast<Policy&&>(policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}

template<class Policy>
auto early_reduce(Policy&& policy, int* first, int* last)
    -> decltype(std::reduce(static_cast<Policy&&>(policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}

#include <hipSYCL/common/filesystem.hpp>
#include <hipSYCL/std/stdpar/detail/sycl_glue.hpp>
#include <execution>
#pragma pop_macro("HIPSYCL_STDPAR_ENTRYPOINT")
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<decltype(early_fill(std::execution::par_unseq,
                                               nullptr, nullptr)), void>);
static_assert(std::is_same_v<decltype(early_reduce(std::execution::par_unseq,
                                                 nullptr, nullptr)), int>);

void check_policy_dispatch(int* first, int* last) {
  using fill_const_object = decltype(std::fill(std::execution::par_unseq, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_const_object = decltype(std::reduce(std::execution::par_unseq, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  static_assert(std::is_same_v<fill_const_object, void>);
  static_assert(std::is_same_v<reduce_const_object, int>);

#ifndef _LIBCPP_VERSION
  // GNU's copyable policies must retain by-value interception for every cvref
  // form, rather than selecting the native ExecutionPolicy&& overload.
  auto policy = std::execution::par_unseq;
  const auto const_policy = std::execution::par_unseq;

  using fill_lvalue = decltype(std::fill(policy, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_lvalue = decltype(std::reduce(policy, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using fill_const_lvalue = decltype(std::fill(const_policy, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_const_lvalue = decltype(std::reduce(const_policy, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using fill_temporary = decltype(std::fill(std::execution::parallel_unsequenced_policy{}, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_temporary = decltype(std::reduce(std::execution::parallel_unsequenced_policy{}, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using fill_rvalue = decltype(std::fill(std::move(policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_rvalue = decltype(std::reduce(std::move(policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using fill_const_rvalue = decltype(std::fill(std::move(const_policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using reduce_const_rvalue = decltype(std::reduce(std::move(const_policy), first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
#endif
}

void check_parallel_dispatch(int* first, int* last, void (*update)(int&)) {
#ifndef _LIBCPP_VERSION
  using for_each_result = decltype(std::for_each(std::execution::par, first, last, update)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
  using fill_result = decltype(std::fill(std::execution::par, first, last, 0)); // expected-warning 1-2 {{AdaptiveCpp stdpar overload selected}}
#else
  // libc++ keeps native par dispatch: an AdaptiveCpp diagnostic is unexpected.
  using for_each_result = decltype(std::for_each(std::execution::par, first, last, update));
  using fill_result = decltype(std::fill(std::execution::par, first, last, 0));
#endif
  static_assert(std::is_same_v<for_each_result, void>);
  static_assert(std::is_same_v<fill_result, void>);
}

using policy = hipsycl::stdpar::par_unseq;
using policy_argument = hipsycl::stdpar::par_unseq_argument;
static_assert(std::is_same_v<policy,
                             std::execution::parallel_unsequenced_policy>);
static_assert(std::is_same_v<policy_argument,
                             std::conditional_t<std::is_copy_constructible_v<policy>,
                                                policy, const policy&>>);

using category = hipsycl::stdpar::algorithm_category::reduce;
using metadata = decltype(hipsycl::stdpar::algorithm{
    category{}, std::execution::par_unseq});
using argument_metadata = decltype(hipsycl::stdpar::algorithm{
    category{}, std::declval<policy_argument>()});
static_assert(std::is_same_v<typename metadata::algorithm_category, category>);
static_assert(std::is_same_v<typename metadata::execution_policy, policy>);
static_assert(std::is_same_v<metadata, argument_metadata>);
