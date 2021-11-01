/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2021 Aksel Alpay
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

#ifndef HIPSYCL_MEMORY_ENVIRONMENT_HPP
#define HIPSYCL_MEMORY_ENVIRONMENT_HPP

#include <type_traits>
#include "backend.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "sp_group.hpp"
#include "sp_private_memory.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {
namespace memory_environment {

template<class First, typename... Rest>
HIPSYCL_UNIVERSAL_TARGET
First get_first(First&& f, Rest&&...) noexcept {
  return f;
}

template<class F, class First, typename... Rest>
HIPSYCL_UNIVERSAL_TARGET
void remove_first(F&& f, First&&, Rest&&... r) noexcept {
  f(r...);
}

template<class First, typename... Args>
HIPSYCL_UNIVERSAL_TARGET
auto get_last(First&& f, Args&&... args) noexcept {
  if constexpr(sizeof...(args) == 0) {
    return f;
  } else {
    return get_last(args...);
  }
}

enum class allocation_type {
 local_mem,
 private_mem
};


template<class T>
struct array_type_traits {
  using scalar_type = T;
};

template<class T, int N>
struct array_type_traits<T [N]> {
  using scalar_type = T;
};

template<class T, int N, int M>
struct array_type_traits<T [N][M]> {
  using scalar_type = T;
};

template<class T, int N, int M, int K>
struct array_type_traits<T [N][M][K]> {
  using scalar_type = T;
};

template<class T>
using array_scalar_t = typename array_type_traits<T>::scalar_type;

template<class T, allocation_type AType, bool IsInitialized>
struct memory_allocation_request {
    using value_type = T;
    static constexpr allocation_type alloc_type = AType;
    static constexpr bool is_initialized = IsInitialized;
};

template<class T, allocation_type AType>
struct memory_allocation_request<T,AType,true> {
    using value_type = T;
    static constexpr allocation_type alloc_type = AType;
    static constexpr bool is_initialized = true;

    array_scalar_t<T> init_value;
};

template<class T>
constexpr std::size_t aligned_alloca_size(std::size_t num_elements) {
  return num_elements * sizeof(T) + sizeof(T) - 1;
}

template<class T>
T* aligned_alloca_offset(void* allocation) {
  std::size_t alloc = reinterpret_cast<size_t>(allocation);

  std::size_t remainder = alloc % alignof(T);
  if(remainder == 0)
    return reinterpret_cast<T*>(alloc);
  
  return reinterpret_cast<T*>(alloc + alignof(T) - alloc % alignof(T));
}

#define HIPSYCL_MAKE_ALIGNED_ALLOCA(T, num_elements, alloc_name)               \
  void *__hipsycl_alloca_allocation##alloc_name =                              \
      alloca(aligned_alloca_size<T>(num_elements));                            \
  T *alloc_name =                                                              \
      aligned_alloca_offset<T>(__hipsycl_alloca_allocation##alloc_name);

} // memory_environment
} // detail

template <class T>
HIPSYCL_KERNEL_TARGET
auto require_local_mem() noexcept {
  using namespace detail::memory_environment;
  return memory_allocation_request<T, allocation_type::local_mem, false>{};
}

template <class T>
HIPSYCL_KERNEL_TARGET auto require_local_mem(
    detail::memory_environment::array_scalar_t<T> init_value) noexcept {
  using namespace detail::memory_environment;
  
  return memory_allocation_request<T, allocation_type::local_mem, true>{
      init_value};
}

template <class T>
HIPSYCL_KERNEL_TARGET
auto require_private_mem() noexcept {
  using namespace detail::memory_environment;
  return memory_allocation_request<T, allocation_type::private_mem, false>{};
}

template <class T>
HIPSYCL_KERNEL_TARGET auto require_private_mem(
    detail::memory_environment::array_scalar_t<T> init_value) noexcept {
  using namespace detail::memory_environment;
  
  return memory_allocation_request<T, allocation_type::private_mem, true>{
      init_value};
}

template <class Group, class FirstArg, typename... RestArgs>
HIPSYCL_KERNEL_TARGET
void memory_environment(const Group &g, FirstArg &&first,
                        RestArgs &&...rest) noexcept {

  using namespace detail::memory_environment;

  if constexpr(sizeof...(RestArgs) == 0) {
    first();
  } else {
    using request_type = FirstArg;
    using value_type = typename request_type::value_type;
    using scalar_type = array_scalar_t<value_type>;
    constexpr size_t num_elements =
            sizeof(value_type) / sizeof(scalar_type);

    auto function = get_last(rest...);
    auto replace_arg = [](auto&& x, auto&& arg, auto&& replacement) {
      if constexpr(std::is_same_v<decltype(x), decltype(arg)>){
        return replacement;
      } else {
        return x;
      }
    };
    
    if constexpr(request_type::alloc_type == 
        allocation_type::local_mem){
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA || HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
      __shared__ value_type memory_declaration;
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
      // TODO
      value_type memory_declaration;
#else
      value_type memory_declaration;
#endif
      if constexpr(request_type::is_initialized) {

        scalar_type* array = reinterpret_cast<scalar_type*>(&memory_declaration);

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
        for(int i = 0; i < num_elements; ++i)
          array[i] = first.init_value;
#else
        const size_t tid = g.get_physical_local_linear_id();
        const size_t lrange = g.get_physical_local_linear_range();
        for(size_t elem = tid; elem < num_elements; elem += lrange) {
          array[elem] = first.init_value;
        }
        group_barrier(g);
#endif
      }
    
      memory_environment(g, replace_arg(rest, function, [&](auto&&... args){
        function(memory_declaration, args...);
      })...);
      
    } else if constexpr(request_type::alloc_type ==
        allocation_type::private_mem){

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
      HIPSYCL_MAKE_ALIGNED_ALLOCA(value_type, 
        g.get_logical_local_linear_range(), memory_declaration_ptr);
#else
      value_type memory_declaration;
      value_type* memory_declaration_ptr = &memory_declaration;
#endif
      if constexpr(request_type::is_initialized) {
        // TODO arrays are not yet supported
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
        for(int i = 0; i < g.get_logical_local_linear_range(); ++i)
          memory_declaration_ptr[i] = first.init_value;
#else
        *memory_declaration_ptr = first.init_value;
#endif
      }

      detail::private_memory_access<value_type, std::decay_t<decltype(g)>>
          mem_access{g, memory_declaration_ptr};
      memory_environment(g, replace_arg(rest, function, [&](auto&&... args){
        function(mem_access, args...);
      })...);
    }
  }
}

template<class T, class Group, class Function>
HIPSYCL_KERNEL_TARGET
void local_memory_environment(const Group& g, Function&& f) noexcept {
  memory_environment(g, require_local_mem<T>(), f);
}

template<class T, class Group, class Function>
HIPSYCL_KERNEL_TARGET
void private_memory_environment(const Group& g, Function&& f) noexcept {
  memory_environment(g, require_private_mem<T>(), f);
}

}
}

#endif
