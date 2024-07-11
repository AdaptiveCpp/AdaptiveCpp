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
ACPP_UNIVERSAL_TARGET
First get_first(First&& f, Rest&&...) noexcept {
  return f;
}

template<class F, class First, typename... Rest>
ACPP_UNIVERSAL_TARGET
void remove_first(F&& f, First&&, Rest&&... r) noexcept {
  f(r...);
}

template<class First, typename... Args>
ACPP_UNIVERSAL_TARGET
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
  void *__acpp_alloca_allocation##alloc_name =                              \
      alloca(aligned_alloca_size<T>(num_elements));                            \
  T *alloc_name =                                                              \
      aligned_alloca_offset<T>(__acpp_alloca_allocation##alloc_name);



template <class Group, class FirstArg, typename... RestArgs>
ACPP_KERNEL_TARGET
void memory_environment_host(const Group &g, FirstArg &&first,
                        RestArgs &&...rest) noexcept {

  if constexpr(sizeof...(RestArgs) == 0) {
    first();
  } else {
    using request_type = std::decay_t<FirstArg>;
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

      value_type memory_declaration;
      if constexpr(request_type::is_initialized) {

        scalar_type* array = reinterpret_cast<scalar_type*>(&memory_declaration);

        for(int i = 0; i < num_elements; ++i)
          array[i] = first.init_value;
      }

      memory_environment_host(g,
                              replace_arg(rest, function, [&](auto &&...args) {
                                function(memory_declaration, args...);
                              })...);

    } else if constexpr(request_type::alloc_type ==
        allocation_type::private_mem){

      HIPSYCL_MAKE_ALIGNED_ALLOCA(value_type, 
        g.get_logical_local_linear_range(), memory_declaration_ptr);

      if constexpr(request_type::is_initialized) {
        for(int i = 0; i < g.get_logical_local_linear_range(); ++i)
          memory_declaration_ptr[i] = first.init_value;
      }

      detail::private_memory_access<value_type, std::decay_t<decltype(g)>>
          mem_access{g, memory_declaration_ptr};
      memory_environment_host(g,
                              replace_arg(rest, function, [&](auto &&...args) {
                                function(mem_access, args...);
                              })...);
    }
  }
}

template <class Group, class FirstArg, typename... RestArgs>
ACPP_KERNEL_TARGET
void memory_environment_device(const Group &g, FirstArg &&first,
                              RestArgs &&...rest) noexcept {

  if constexpr(sizeof...(RestArgs) == 0) {
    first();
  } else {
    using request_type = std::decay_t<FirstArg>;
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
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA || ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
      __shared__ value_type memory_declaration;
#else // SSCP
      // TODO
      value_type memory_declaration;
#endif
      if constexpr(request_type::is_initialized) {

        scalar_type* array = reinterpret_cast<scalar_type*>(&memory_declaration);

        const size_t tid = g.get_physical_local_linear_id();
        const size_t lrange = g.get_physical_local_linear_range();
        for(size_t elem = tid; elem < num_elements; elem += lrange) {
          array[elem] = first.init_value;
        }
        group_barrier(g);
        
      }

      memory_environment_device(
          g, replace_arg(rest, function, [&](auto &&...args) {
            function(memory_declaration, args...);
          })...);

    } else if constexpr(request_type::alloc_type ==
        allocation_type::private_mem){

      value_type memory_declaration;

      if constexpr(request_type::is_initialized) {
        // TODO arrays are not yet supported
        memory_declaration = first.init_value;
      }

      detail::private_memory_access<value_type, std::decay_t<decltype(g)>>
          mem_access{g, &memory_declaration};

      memory_environment_device(
          g, replace_arg(rest, function, [&](auto &&...args) {
            function(mem_access, args...);
          })...);
    }
  }
}


} // memory_environment
} // detail

template <class T>
ACPP_KERNEL_TARGET
auto require_local_mem() noexcept {
  using namespace detail::memory_environment;
  return memory_allocation_request<T, allocation_type::local_mem, false>{};
}

template <class T>
ACPP_KERNEL_TARGET auto require_local_mem(
    detail::memory_environment::array_scalar_t<T> init_value) noexcept {
  using namespace detail::memory_environment;
  
  return memory_allocation_request<T, allocation_type::local_mem, true>{
      init_value};
}

template <class T>
ACPP_KERNEL_TARGET
auto require_private_mem() noexcept {
  using namespace detail::memory_environment;
  return memory_allocation_request<T, allocation_type::private_mem, false>{};
}

template <class T>
ACPP_KERNEL_TARGET auto require_private_mem(
    detail::memory_environment::array_scalar_t<T> init_value) noexcept {
  using namespace detail::memory_environment;
  
  return memory_allocation_request<T, allocation_type::private_mem, true>{
      init_value};
}

template <class Group, class FirstArg, typename... RestArgs>
ACPP_KERNEL_TARGET
void memory_environment(const Group &g, FirstArg &&first,
                        RestArgs &&...rest) noexcept {
  __acpp_if_target_device(
    detail::memory_environment::memory_environment_device(g, first, rest...);
  );
  __acpp_if_target_host(
    detail::memory_environment::memory_environment_host(g, first, rest...);
  );
}

template<class T, class Group, class Function>
ACPP_KERNEL_TARGET
void local_memory_environment(const Group& g, Function&& f) noexcept {
  memory_environment(g, require_local_mem<T>(), f);
}

template<class T, class Group, class Function>
ACPP_KERNEL_TARGET
void private_memory_environment(const Group& g, Function&& f) noexcept {
  memory_environment(g, require_private_mem<T>(), f);
}

}
}

#endif
