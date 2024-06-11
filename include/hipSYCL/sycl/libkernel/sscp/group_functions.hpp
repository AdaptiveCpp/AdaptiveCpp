/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021-2022 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include "../marray.hpp"
#include "../bit_cast.hpp"
#include "../half.hpp"
#include <limits>
#include <type_traits>

/// TODO: This file is a placeholder, most group algorithms are unimplemented!

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP

#include "builtins/barrier.hpp"
#include "builtins/broadcast.hpp"
#include "builtins/collpredicate.hpp"
#include "builtins/reduction.hpp"
#include "builtins/shuffle.hpp"

namespace hipsycl {
namespace sycl::detail::sscp_builtins {

template<class Tout, class Tin>
HIPSYCL_FORCE_INLINE
Tout maybe_bit_cast(Tin x) {
  static_assert(sizeof(Tout) == sizeof(Tin), "Invalid data type");
  if constexpr(std::is_same_v<Tin, Tout>) {
    return x;
  } else {
    return bit_cast<Tout>(x);
  }
}

// barrier
template <int Dim>
HIPSYCL_BUILTIN void
__acpp_group_barrier(group<Dim> g,
                        memory_scope fence_scope = group<Dim>::fence_scope) {
  __acpp_sscp_work_group_barrier(fence_scope, memory_order::seq_cst);
}

HIPSYCL_BUILTIN
 void
__acpp_group_barrier(sub_group g,
                        memory_scope fence_scope = sub_group::fence_scope) {
  __acpp_sscp_sub_group_barrier(fence_scope, memory_order::seq_cst);
}

// broadcast

template <int Dim, typename T>
HIPSYCL_BUILTIN 
T __acpp_group_broadcast(
    group<Dim> g, T x,
    typename group<Dim>::linear_id_type local_linear_id = 0) {

  
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_broadcast_i8(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_broadcast_i16(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_broadcast_i32(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_broadcast_i64(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int64>(x)));
  }
}

template <int Dim, typename T>
HIPSYCL_BUILTIN T __acpp_group_broadcast(
    group<Dim> g, T x, typename group<Dim>::id_type local_id) {

  const auto sender_lid = linear_id<g.dimensions>::get(
      local_id, g.get_local_range());
  return __acpp_group_broadcast(g, x, static_cast<int>(sender_lid));
}

template <typename T>
HIPSYCL_BUILTIN T __acpp_group_broadcast(
    sub_group g, T x, typename sub_group::linear_id_type local_linear_id = 0) {

  // Song recommendation: Leaves' Eyes - Angel and the Ghost
  //
  // Ring out the bells, let my mourning echo through the walls
  // My heart sleeps so lonely, my ghost will seek you at night!
  
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_broadcast_i8(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_broadcast_i16(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_broadcast_i32(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_broadcast_i64(
        static_cast<__acpp_int32>(local_linear_id),
        maybe_bit_cast<__acpp_int64>(x)));
  }
}

template<typename T>
HIPSYCL_BUILTIN
T __acpp_group_broadcast(sub_group g, T x,
                  typename sub_group::id_type local_id) {

  return __acpp_group_broadcast(g, x, static_cast<int>(local_id[0]));
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(vec<T, N>) > 8), vec<T,N>> 
__acpp_group_broadcast(
    Group g, vec<T, N> x,
    typename Group::linear_id_type local_linear_id = 0) {
  vec<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = __acpp_group_broadcast(g, x[i], local_linear_id);
  }
  return result;
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(marray<T, N>) > 8), marray<T,N>>
__acpp_group_broadcast(
    Group g, marray<T, N> x,
    typename Group::linear_id_type local_linear_id = 0) {
  marray<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = __acpp_group_broadcast(g, x[i], local_linear_id);
  }
  return result;
}


// any_of


template<int Dim>
HIPSYCL_BUILTIN
bool __acpp_any_of_group(group<Dim> g, bool pred) {
  return __acpp_sscp_work_group_any(pred);
}

HIPSYCL_BUILTIN
bool __acpp_any_of_group(sub_group g, bool pred) {
  return __acpp_sscp_sub_group_any(pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN 
bool __acpp_joint_any_of(Group g, Ptr first, Ptr last,
                            Predicate pred) {
  // This is copy-pasted from the hiplike implementation. Can
  // we share/unify this code somehow?
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = false;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local |= pred(*p);

  return __acpp_any_of_group(g, local);
}

// all_of


template<int Dim>
HIPSYCL_BUILTIN
bool __acpp_all_of_group(group<Dim> g, bool pred) {
  return __acpp_sscp_work_group_all(pred);
}

HIPSYCL_BUILTIN
bool __acpp_all_of_group(sub_group g, bool pred) {
  return __acpp_sscp_sub_group_all(pred);
}


template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN bool __acpp_joint_all_of(Group g, Ptr first, Ptr last,
                                                  Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = true;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local &= pred(*p);

  return __acpp_all_of_group(g, local);
}

// none_of

template<int Dim>
HIPSYCL_BUILTIN
bool __acpp_none_of_group(group<Dim> g, bool pred) {
  return __acpp_sscp_work_group_none(pred);
}

HIPSYCL_BUILTIN
bool __acpp_none_of_group(sub_group g, bool pred) {
  return __acpp_sscp_sub_group_none(pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN 
bool __acpp_joint_none_of(Group g, Ptr first, Ptr last,
                             Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = false;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local |= pred(*p);

  return __acpp_none_of_group(g, local);
}

// reduce

template<class Op>
struct sscp_binary_operation {};

#define HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(Op, SSCPOp)                    \
  template <class T> struct sscp_binary_operation<Op<T>> {                     \
    static constexpr __acpp_sscp_algorithm_op value = SSCPOp;               \
  };

HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::plus,
                                        __acpp_sscp_algorithm_op::plus)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::multiplies,
                                        __acpp_sscp_algorithm_op::multiply)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::minimum,
                                        __acpp_sscp_algorithm_op::min)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::maximum,
                                        __acpp_sscp_algorithm_op::max)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::bit_and,
                                        __acpp_sscp_algorithm_op::bit_and)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::bit_or,
                                        __acpp_sscp_algorithm_op::bit_or)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::bit_xor,
                                        __acpp_sscp_algorithm_op::bit_xor)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::logical_and,
                                        __acpp_sscp_algorithm_op::logical_and)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(sycl::logical_or,
                                        __acpp_sscp_algorithm_op::logical_or)

HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::plus,
                                        __acpp_sscp_algorithm_op::plus)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::multiplies,
                                        __acpp_sscp_algorithm_op::multiply)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::bit_and,
                                        __acpp_sscp_algorithm_op::bit_and)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::bit_or,
                                        __acpp_sscp_algorithm_op::bit_or)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::bit_xor,
                                        __acpp_sscp_algorithm_op::bit_xor)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::logical_and,
                                        __acpp_sscp_algorithm_op::logical_and)
HIPSYCL_SSCP_MAP_GROUP_BINARY_OPERATION(std::logical_or,
                                        __acpp_sscp_algorithm_op::logical_or)

template <class Op>
inline constexpr __acpp_sscp_algorithm_op sscp_binary_operation_v =
    sscp_binary_operation<Op>::value;

template<class T, __acpp_sscp_algorithm_op Sscp_op>
struct sscp_binary_operation_identity {};

#define HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(SSCPOp, Identity)               \
  template <class T> struct sscp_binary_operation_identity<T, SSCPOp> {        \
    static auto get() { return Identity; }                                     \
  };                                                                           \
  template <class T, int N>                                                    \
  struct sscp_binary_operation_identity<vec<T, N>, SSCPOp> {                   \
    static auto get() { return vec<T, N>{Identity}; }                          \
  };                                                                           \
  template <class T, int N>                                                    \
  struct sscp_binary_operation_identity<marray<T, N>, SSCPOp> {                \
    static auto get() { return marray<T, N>{Identity}; }                       \
  };

HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::plus, T{0})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::multiply, T{1})
// TODO This is not really correct for floating point - those should use infinity. But then, what about
// compilation with -ffast-math?
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::min, T{std::numeric_limits<T>::max()})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::max, T{std::numeric_limits<T>::min()})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::bit_and, ~T{0})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::bit_or, T{0})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::bit_xor, T{0})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::logical_and, T{1})
HIPSYCL_SSCP_MAP_GROUP_BINARY_IDENTITY(__acpp_sscp_algorithm_op::logical_or, T{0})

template <
    typename T, typename BinaryOperation,
    std::enable_if_t<(std::is_integral_v<T> && std::is_signed_v<T>), int> = 0>
HIPSYCL_BUILTIN T __acpp_reduce_over_group(sub_group g, T x,
                                              BinaryOperation binary_op) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_i8(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_i16(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_i32(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_i64(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int64>(x)));
  }
}

template <
    typename T, typename BinaryOperation,
    std::enable_if_t<(std::is_integral_v<T> && !std::is_signed_v<T>), int> = 0>
HIPSYCL_BUILTIN T __acpp_reduce_over_group(sub_group g, T x,
                                              BinaryOperation binary_op) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_u8(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_u16(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_u32(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_reduce_u64(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint64>(x)));
  }
}

template<typename BinaryOperation>
HIPSYCL_BUILTIN
half __acpp_reduce_over_group(sub_group g, half x, BinaryOperation binary_op) {
  return detail::create_half(__acpp_sscp_sub_group_reduce_f16(
      sscp_binary_operation_v<BinaryOperation>, detail::get_half_storage(x)));
}

template<typename BinaryOperation>
HIPSYCL_BUILTIN
float __acpp_reduce_over_group(sub_group g, float x, BinaryOperation binary_op) {
  return __acpp_sscp_sub_group_reduce_f32(
      sscp_binary_operation_v<BinaryOperation>, x);
}

template<typename BinaryOperation>
HIPSYCL_BUILTIN
double __acpp_reduce_over_group(sub_group g, double x, BinaryOperation binary_op) {
  return __acpp_sscp_sub_group_reduce_f64(
      sscp_binary_operation_v<BinaryOperation>, x);
}

template<typename T, int N, typename BinaryOperation>
HIPSYCL_BUILTIN
vec<T,N> __acpp_reduce_over_group(sub_group g, vec<T,N> x, BinaryOperation binary_op) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_reduce_over_group(g, x[i], binary_op);
  }
  return result;
}

template<typename T, int N, typename BinaryOperation>
HIPSYCL_BUILTIN
marray<T,N> __acpp_reduce_over_group(sub_group g, marray<T,N> x, BinaryOperation binary_op) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_reduce_over_group(g, x[i], binary_op);
  }
  return result;
}


template <
    typename T, int Dim, typename BinaryOperation,
    std::enable_if_t<(std::is_integral_v<T> && std::is_signed_v<T>), int> = 0>
HIPSYCL_BUILTIN
T __acpp_reduce_over_group(group<Dim> g, T x, BinaryOperation binary_op) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_i8(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_i16(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_i32(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_i64(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_int64>(x)));
  }
}

template <
    typename T, int Dim, typename BinaryOperation,
    std::enable_if_t<(std::is_integral_v<T> && !std::is_signed_v<T>), int> = 0>
HIPSYCL_BUILTIN T __acpp_reduce_over_group(group<Dim> g, T x,
                                              BinaryOperation binary_op) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_u8(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint8>(x)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_u16(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint16>(x)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_u32(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint32>(x)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_reduce_u64(
        sscp_binary_operation_v<BinaryOperation>,
        maybe_bit_cast<__acpp_uint64>(x)));
  }
}

template<int Dim, typename BinaryOperation>
HIPSYCL_BUILTIN
half __acpp_reduce_over_group(group<Dim> g, half x, BinaryOperation binary_op) {
  return detail::create_half(__acpp_sscp_work_group_reduce_f16(
      sscp_binary_operation_v<BinaryOperation>, detail::get_half_storage(x)));
}

template<int Dim, typename BinaryOperation>
HIPSYCL_BUILTIN
float __acpp_reduce_over_group(group<Dim> g, float x, BinaryOperation binary_op) {
  return __acpp_sscp_work_group_reduce_f32(
      sscp_binary_operation_v<BinaryOperation>, x);
}

template<int Dim, typename BinaryOperation>
HIPSYCL_BUILTIN
double __acpp_reduce_over_group(group<Dim> g, double x, BinaryOperation binary_op) {
  return __acpp_sscp_work_group_reduce_f64(
      sscp_binary_operation_v<BinaryOperation>, x);
}


template<typename T, int N, int Dim, typename BinaryOperation>
HIPSYCL_BUILTIN
vec<T,N> __acpp_reduce_over_group(group<Dim> g, vec<T,N> x, BinaryOperation binary_op) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_reduce_over_group(g, x[i], binary_op);
  }
  return result;
}

template<typename T, int N, int Dim, typename BinaryOperation>
HIPSYCL_BUILTIN
marray<T,N> __acpp_reduce_over_group(group<Dim> g, marray<T,N> x, BinaryOperation binary_op) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_reduce_over_group(g, x[i], binary_op);
  }
  return result;
}

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
typename std::iterator_traits<Ptr>::value_type
__acpp_joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  
  const size_t lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t lid          = g.get_local_linear_id();

  using value_type = std::remove_reference_t<decltype(*first)>;

  if(num_elements == 0)
    return value_type{};
  
  if(num_elements == 1)
    return *first;
  
  Ptr start_ptr = first + lid;

  using type = decltype(*first);

  auto local = sscp_binary_operation_identity<
      std::decay_t<type>, sscp_binary_operation_v<BinaryOperation>>::get();
  if(start_ptr < last)
    local = *start_ptr;
  
  for (Ptr p = start_ptr + lrange; p < last; p += lrange)
    local = binary_op(local, *p);

  return __acpp_reduce_over_group(g, local, binary_op);
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T __acpp_joint_reduce(Group g, Ptr first, Ptr last, T init,
                         BinaryOperation binary_op) {
  return binary_op(__acpp_joint_reduce(g, first, last, binary_op), init);
}

// exclusive_scan


template <int Dim, typename V, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __acpp_exclusive_scan_over_group(
    group<Dim> g, V x, T init, BinaryOperation binary_op);


template <typename V, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __acpp_exclusive_scan_over_group(
    sub_group g, V x, T init, BinaryOperation binary_op);

template <typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN T
__acpp_exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op);

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__acpp_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               T init, BinaryOperation binary_op);

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__acpp_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op);

// inclusive_scan

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__acpp_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op, T init);

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__acpp_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op);

template <int Dim, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN
T __acpp_inclusive_scan_over_group(
    group<Dim> g, T x, BinaryOperation binary_op);

template <typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __acpp_inclusive_scan_over_group(
    sub_group g, T x, BinaryOperation binary_op);

template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN T __acpp_inclusive_scan_over_group(
    Group g, V x, T init, BinaryOperation binary_op) {
  auto scan = __acpp_inclusive_scan_over_group(g, T{x}, binary_op);
  return binary_op(scan, init);
}

// shift_left
template <int Dim, typename T>
HIPSYCL_BUILTIN
T __acpp_shift_group_left(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {

  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shl_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shl_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shl_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_uint32>(delta)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shl_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_uint32>(delta)));
  }
}

template <typename T>
HIPSYCL_BUILTIN T __acpp_shift_group_left(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1) {

  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shl_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shl_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shl_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_uint32>(delta)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shl_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_uint32>(delta)));
  }
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(vec<T, N>) > 8), vec<T,N>>
__acpp_shift_group_left(Group g, vec<T,N> x, typename Group::linear_id_type delta = 1) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_shift_group_left(g, x[i], delta);
  }
  return result;
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(marray<T, N>) > 8), marray<T,N>>
__acpp_shift_group_left(Group g, marray<T,N> x, typename Group::linear_id_type delta = 1) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_shift_group_left(g, x[i], delta);
  }
  return result;
}

// shift_right
template <int Dim, typename T>
HIPSYCL_BUILTIN T __acpp_shift_group_right(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shr_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shr_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shr_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_uint32>(delta)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_shr_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_uint32>(delta)));
  }
}

template <typename T>
HIPSYCL_BUILTIN T __acpp_shift_group_right(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shr_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shr_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_uint32>(delta)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shr_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_uint32>(delta)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_shr_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_uint32>(delta)));
  }
}


template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(vec<T, N>) > 8), vec<T,N>>
__acpp_shift_group_right(Group g, vec<T,N> x, typename Group::linear_id_type delta = 1) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_shift_group_right(g, x[i], delta);
  }
  return result;
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(marray<T, N>) > 8), marray<T,N>>
__acpp_shift_group_right(Group g, marray<T,N> x, typename Group::linear_id_type delta = 1) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_shift_group_right(g, x[i], delta);
  }
  return result;
}

// permute_group_by_xor
template <int Dim, typename T>
HIPSYCL_BUILTIN T __acpp_permute_group_by_xor(
    group<Dim> g, T x, typename group<Dim>::linear_id_type mask) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_permute_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_int32>(mask)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_permute_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_int32>(mask)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_permute_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_int32>(mask)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_permute_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_int32>(mask)));
  }
}

template <typename T>
HIPSYCL_BUILTIN T __acpp_permute_group_by_xor(
    sub_group g, T x, typename sub_group::linear_id_type mask) {
  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_permute_i8(
        maybe_bit_cast<__acpp_int8>(x), static_cast<__acpp_int32>(mask)));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_permute_i16(
        maybe_bit_cast<__acpp_int16>(x), static_cast<__acpp_int32>(mask)));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_permute_i32(
        maybe_bit_cast<__acpp_int32>(x), static_cast<__acpp_int32>(mask)));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_permute_i64(
        maybe_bit_cast<__acpp_int64>(x), static_cast<__acpp_int32>(mask)));
  }
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(vec<T, N>) > 8), vec<T,N>>
__acpp_permute_group_by_xor(Group g, vec<T,N> x, typename Group::linear_id_type mask) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_permute_group_by_xor(g, x[i], mask);
  }
  return result;
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(marray<T, N>) > 8), marray<T,N>>
__acpp_permute_group_by_xor(Group g, marray<T,N> x, typename Group::linear_id_type mask) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_permute_group_by_xor(g, x[i], mask);
  }
  return result;
}

// select_from_group
template <int Dim, typename T>
HIPSYCL_BUILTIN T __acpp_select_from_group(
    group<Dim> g, T x, typename group<Dim>::id_type remote_local_id) {

  __acpp_int32 linear_id = static_cast<__acpp_int32>(
      sycl::detail::linear_id<g.dimensions>::get(remote_local_id, g.get_local_range()));

  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_select_i8(
        maybe_bit_cast<__acpp_int8>(x), linear_id));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_select_i16(
        maybe_bit_cast<__acpp_int16>(x), linear_id));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_select_i32(
        maybe_bit_cast<__acpp_int32>(x), linear_id));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_work_group_select_i64(
        maybe_bit_cast<__acpp_int64>(x), linear_id));
  }
}

template <typename T>
HIPSYCL_BUILTIN T __acpp_select_from_group(
    sub_group g, T x, typename sub_group::id_type remote_local_id) {

  __acpp_int32 linear_id = static_cast<__acpp_int32>(remote_local_id[0]);

  if constexpr(sizeof(T) == 1) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_select_i8(
        maybe_bit_cast<__acpp_int8>(x), linear_id));
  } else if constexpr(sizeof(T) == 2) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_select_i16(
        maybe_bit_cast<__acpp_int16>(x), linear_id));
  } else if constexpr(sizeof(T) == 4) {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_select_i32(
        maybe_bit_cast<__acpp_int32>(x), linear_id));
  } else {
    return maybe_bit_cast<T>(__acpp_sscp_sub_group_select_i64(
        maybe_bit_cast<__acpp_int64>(x), linear_id));
  }
}


template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(vec<T, N>) > 8), vec<T,N>>
__acpp_select_from_group(Group g, vec<T,N> x, typename Group::id_type remote_local_id) {
  vec<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_select_from_group(g, x[i], remote_local_id);
  }
  return result;
}

template <class Group, typename T, int N>
HIPSYCL_BUILTIN
std::enable_if_t<(sizeof(marray<T, N>) > 8), marray<T,N>>
__acpp_select_from_group(Group g, marray<T,N> x, typename Group::id_type remote_local_id) {
  marray<T,N> result;
  for(int i = 0; i < N; ++i) {
    result[i] = __acpp_select_from_group(g, x[i], remote_local_id);
  }
  return result;
}

} // namespace sycl::detail::sscp_builtins
} // namespace hipsycl

#endif

#endif // HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP

