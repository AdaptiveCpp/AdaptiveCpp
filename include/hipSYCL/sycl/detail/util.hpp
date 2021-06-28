/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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
  
  static_assert(
      sizeof...(args) > 0,
      "Cannot extract last argument from template pack for empty pack");
  
  constexpr std::size_t last_index = sizeof...(args) - 1;

  auto last_element =
      std::get<last_index>(std::forward_as_tuple(std::forward<Args>(args)...));

  auto preceding_elements =
      extract_tuple(std::forward_as_tuple(std::forward<Args>(args)...),
                    std::make_index_sequence<last_index>());

  std::apply(f,
             std::tuple_cat(std::make_tuple(last_element), preceding_elements));
}

template <class Tout, class Tin>
Tout bit_cast(Tin x) {
  static_assert(sizeof(Tout)==sizeof(Tin), "Types must match sizes");

  Tout out;
#if !defined(__APPLE__) && defined(__clang_major__) && __clang_major__ >= 11
  __builtin_memcpy_inline(&out, &x, sizeof(Tin));
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
  memcpy(&out, &x, sizeof(Tin));
#else
  char* cout = &out;
  char* cin = &x;
  for(int i = 0; i < sizeof(Tin); ++i)
    cout[i] = cin[i];
#endif
  
  return out;
}

}
}
}

#endif

