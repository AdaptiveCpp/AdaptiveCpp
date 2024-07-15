/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_UTIL_HPP
#define HIPSYCL_UTIL_HPP

#include <atomic>
#include <tuple>
#include <utility>
#include <cstring>
#include "hipSYCL/sycl/libkernel/backend.hpp"


namespace hipsycl {
namespace sycl {
namespace detail {

template <typename T>
T* get_raw_pointer(T* ptr) { return ptr; }

template <typename WrappedPtr>
auto get_raw_pointer(const WrappedPtr& ptr) { return ptr.get(); }

class spin_lock {
public:
  void lock() {
    while (_lock.test_and_set(std::memory_order_acquire));
  }
  void unlock() {
    _lock.clear(std::memory_order_release);
  }
private:
  std::atomic_flag _lock = ATOMIC_FLAG_INIT;
};

class spin_lock_guard {
public:
  spin_lock_guard(spin_lock& lock) : _lock(lock) {
    _lock.lock();
  }
  ~spin_lock_guard() {
    _lock.unlock();
  }
private:
  spin_lock& _lock;
};

template<typename Tuple, std::size_t... Ints>
std::tuple<std::tuple_element_t<Ints, Tuple>...>
extract_tuple(Tuple&& tuple, std::index_sequence<Ints...>) {
 return { std::get<Ints>(std::forward<Tuple>(tuple))... };
}


template<class F, typename... Args>
void separate_last_argument_and_apply(F&& f, Args&& ... args) {
  
  if constexpr(sizeof...(args) > 0) {
    
    constexpr std::size_t last_index = sizeof...(args) - 1;

    auto last_element =
        std::get<last_index>(std::forward_as_tuple(std::forward<Args>(args)...));

    auto preceding_elements =
        extract_tuple(std::forward_as_tuple(std::forward<Args>(args)...),
                      std::make_index_sequence<last_index>());

    std::apply(f,
              std::tuple_cat(std::make_tuple(last_element), preceding_elements));
  } else {
    // We still need the if constexpr, because otherwise parsing may
    // not terminate in case of an empty argument pack, and the
    // static_assert is never evaluated.
    static_assert(sizeof...(args) > 0,
                  "Invalid call with empty argument list");
  }
}


}
}
}

#endif

