/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {
// reduce implementation
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op, T *scratch) {
  const size_t lid = g.get_local_linear_id();

  scratch[lid] = x;
  group_barrier(g);

  if (g.leader()) {
    T result = scratch[0];

    for (int i = 1; i < g.get_local_range().size(); ++i)
      result = binary_op(result, scratch[i]);

    scratch[0] = result;
  }

  group_barrier(g);
  T tmp = scratch[0];
  group_barrier(g);

  return tmp;
}

// functions using pointers
// any_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, Ptr first, Ptr last) {
  bool result = false;

  if (g.leader()) {
    while (first < last) {
      if (*(first++)) {
        result = true;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = false;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = true;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, Ptr first, Ptr last) {
  const bool result = leader_any_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = leader_any_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// all_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, Ptr first, Ptr last) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (!*(first++)) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (!pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, Ptr first, Ptr last) {
  const bool result = leader_all_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = leader_all_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// none_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, Ptr first, Ptr last) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (*(first++)) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, Ptr first, Ptr last) {
  auto result = leader_none_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  auto result = leader_none_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// reduce
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  T result{};

  if (first >= last) {
    return T{};
  }

  if (g.leader()) {
#pragma omp simd
    for (T *i = first; i < last; ++i)
      result = binary_op(result, *i);
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, V init, BinaryOperation binary_op) {
  auto result = leader_reduce(g, first, last, binary_op);

  if (g.leader()) {
    result = binary_op(result, init);
  }
  return result;
}

template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  T result{};

  if (first >= last) {
    return T{};
  }

  if (g.leader()) {
    result = *(first++);
    while (first != last)
      result = binary_op(result, *(first++));
  }
  return group_broadcast(g, result);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, V *first, V *last, T init, BinaryOperation binary_op) {
  const auto result = leader_reduce(g, first, last, init, binary_op);

  return group_broadcast(g, result);
}

// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {

  if (g.leader()) {
    *(result++) = init;
    while (first != last - 1) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_exclusive_scan(g, first, last, result, T{}, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
  const auto ret = leader_exclusive_scan(g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, T{}, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {
  if (first == last)
    return result;

  if (g.leader()) {
    *(result++) = binary_op(init, *(first++));
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_inclusive_scan(g, first, last, result, T{}, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
  auto ret = leader_inclusive_scan(g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, T{}, binary_op);
}

} // namespace detail

// broadcast
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  if (lid == local_linear_id) {
    scratch[0] = x;
  }

  group_barrier(g);
  T tmp = scratch[0];
  group_barrier(g);

  return tmp;
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  const size_t target_lid =
      detail::linear_id<g.dimensions>::get(local_id, g.get_local_range());
  return group_broadcast(g, x, target_lid);
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
  return x;
}

// barrier
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(Group g, memory_scope fence_scope = Group::fence_scope) {
  if (fence_scope == memory_scope::work_item) {
    // doesn't need sync
  } else if (fence_scope == memory_scope::sub_group) {
    // doesn't need sync (sub_group size = 1)
  } else if (fence_scope == memory_scope::work_group) {
    g.barrier();
  } else if (fence_scope == memory_scope::device) {
    g.barrier();
  }
}

template<>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(sub_group g, memory_scope fence_scope) {
  // doesn't need sync
}

// any_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = false;
  group_barrier(g);

  if (pred)
    scratch[0] = pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(sub_group g, bool pred) {
  return pred;
}

// all_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = true;
  group_barrier(g);

  if (!pred)
    scratch[0] = pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(sub_group g, bool pred) {
  return pred;
}

// none_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = true;
  group_barrier(g);

  if (pred)
    scratch[0] = !pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(sub_group g, bool pred) {
  return pred;
}

// reduce
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  T tmp = detail::group_reduce(g, x, binary_op, scratch);

  return tmp;
}

template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(sub_group g, T x, BinaryOperation binary_op) {
  return x;
}

// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  if (lid + 1 < 1024)
    scratch[lid + 1] = x;
  group_barrier(g);

  if (g.leader()) {
    scratch[0] = init;
    for (int i = 1; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1]);
  }

  group_barrier(g);
  T tmp = scratch[lid];
  group_barrier(g);

  return tmp;
}

template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(sub_group g, V x, T init, BinaryOperation binary_op) {
  return binary_op(x, init);
}

// inclusive_scan
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  scratch[lid] = x;
  group_barrier(g);

  if (g.leader()) {
    for (int i = 1; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1]);
  }

  group_barrier(g);
  T tmp = scratch[lid];
  group_barrier(g);

  return tmp;
}

template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(sub_group g, T x, BinaryOperation binary_op) {
  return x;
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
