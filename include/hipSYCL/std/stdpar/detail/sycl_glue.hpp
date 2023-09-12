/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_PSTL_SYCL_GLUE_HPP
#define HIPSYCL_PSTL_SYCL_GLUE_HPP



#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include <cstdlib>
#include <hipSYCL/sycl/queue.hpp>
#include <hipSYCL/sycl/device.hpp>
#include <hipSYCL/sycl/context.hpp>
#include <hipSYCL/sycl/usm.hpp>
#include <hipSYCL/sycl/usm_query.hpp>
// Fetch builtin declarations to aid SSCP StdBuiltinRemapperPass for
// std:: math function support inside kernels.
#include <hipSYCL/sycl/libkernel/builtin_interface.hpp>
#include <new>

extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void*);

namespace hipsycl::stdpar::detail {

inline sycl::queue construct_default_queue() {
  return sycl::queue{hipsycl::sycl::property_list{
        hipsycl::sycl::property::queue::in_order{},
        hipsycl::sycl::property::queue::hipSYCL_coarse_grained_events{}}};
}

class stdpar_tls_runtime {
private:
  stdpar_tls_runtime()
      : _queue{construct_default_queue()},
        _device_scratch_cache{algorithms::util::allocation_type::device},
        _shared_scratch_cache{algorithms::util::allocation_type::shared},
        _host_scratch_cache{algorithms::util::allocation_type::host} {}

  ~stdpar_tls_runtime() {
    _device_scratch_cache.purge();
    _shared_scratch_cache.purge();
    _host_scratch_cache.purge();
  }

  sycl::queue _queue;
  algorithms::util::allocation_cache _device_scratch_cache;
  algorithms::util::allocation_cache _shared_scratch_cache;
  algorithms::util::allocation_cache _host_scratch_cache;
public:
  
  sycl::queue& get_queue() {
    return _queue;
  }

  template<algorithms::util::allocation_type AT>
  algorithms::util::allocation_cache& get_scratch_cache() {
    if constexpr(AT == algorithms::util::allocation_type::device)
      return _device_scratch_cache;
    else if constexpr(AT == algorithms::util::allocation_type::shared)
      return _shared_scratch_cache;
    else
      return _host_scratch_cache;
  }

  template<algorithms::util::allocation_type AT>
  algorithms::util::allocation_group make_scratch_group() {
    algorithms::util::allocation_cache& cache = get_scratch_cache<AT>();
    return algorithms::util::allocation_group{
        &cache, get_queue().get_device().hipSYCL_device_id()};
  }

  static stdpar_tls_runtime& get() {
    static thread_local stdpar_tls_runtime rt;
    return rt;
  }
};

class single_device_dispatch {
public:
  static hipsycl::sycl::queue& get_queue() {
    return stdpar_tls_runtime::get().get_queue();
  }
};

}

#if defined(__clang__) && defined(HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST) &&    \
    !defined(__HIPSYCL_STDPAR_ASSUME_SYSTEM_USM__)

namespace hipsycl::stdpar::detail {

struct usm_context {
  usm_context() {
    _ctx = single_device_dispatch::get_queue().get_context();
    _is_alive = true;
  }

  ~usm_context() {
    _is_alive = false;
  }

  static bool is_alive() {
    return _is_alive;
  }

  hipsycl::sycl::context& get() {
    return _ctx;
  }
private:
  hipsycl::sycl::context _ctx;
  inline static std::atomic<bool> _is_alive;
};

}


namespace hipsycl::stdpar {

class unified_shared_memory {
public:

  static void pop_disabled() {
    thread_local_storage::get().disabled_stack--;
  }

  static void push_disabled() {
    thread_local_storage::get().disabled_stack++;
  }
  
  static void* malloc(std::size_t n, std::size_t alignment = 0) {
    // Seems some apps rely on n==0 still returning a valid pointer
    if(n == 0)
      n = 1;
    
    if(thread_local_storage::get().disabled_stack == 0) {
      
      void* ptr = nullptr;
      push_disabled();
      if (alignment != 0) {
        ptr = sycl::aligned_alloc_shared(alignment, n,
                                          detail::single_device_dispatch::get_queue());
      } else {
        ptr = sycl::malloc_shared(n, detail::single_device_dispatch::get_queue());
      }
      get()._is_initialized = true;
      pop_disabled();
      return ptr;

    } else {
      if(alignment != 0) {
        void *ptr = 0;
        posix_memalign(&ptr, alignment, n);
        return ptr;
      } else
        return __libc_malloc(n);
    }
  }

  static void free(void* ptr) {
    if(!get()._is_initialized) {
      __libc_free(ptr);
      return;
    }

    if (thread_local_storage::get().disabled_stack == 0) {
          
      push_disabled();
      static detail::usm_context ctx;
      pop_disabled();

      if(!detail::usm_context::is_alive())
        // If the runtime has already shut down, we cannot really
        // reliably free things anymore :( Currently, we just ignore
        // the request.
        return;

      push_disabled();
      if (hipsycl::sycl::get_pointer_type(ptr, ctx.get()) ==
          hipsycl::sycl::usm::alloc::unknown) {
        __libc_free(ptr);
      } else {
        sycl::free(ptr, ctx.get());
      }
      pop_disabled();
    } else {
      __libc_free(ptr);
    }
  }
private:
  unified_shared_memory()
  : _is_initialized{false} {}

  static unified_shared_memory& get() {
    static unified_shared_memory usm_state;
    return usm_state;
  }

  std::atomic<bool> _is_initialized;

  class thread_local_storage {
  public:
    static thread_local_storage& get() {
      static thread_local thread_local_storage s;
      return s;
    }

#ifdef HIPSYCL_STDPAR_MEMORY_MANAGEMENT_DEFAULT_DISABLED
    int disabled_stack = 1;
#else
    int disabled_stack = 0;
#endif
  private:
    thread_local_storage(){}
  };
};

}


#define HIPSYCL_STDPAR_ALLOC [[clang::annotate("hipsycl_stdpar_alloc")]] __attribute__((noinline))
#define HIPSYCL_STDPAR_FREE [[clang::annotate("hipsycl_stdpar_free")]]



HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr)
    throw std::bad_alloc{};
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, const std::nothrow_t&) noexcept  {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::size_t sz ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::size_t sz ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::size_t sz,
                        std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::size_t sz,
                        std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::align_val_t al,
                        const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::align_val_t al,
                        const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

/* Both libc++ and libstdc++ define std::malloc as ::malloc and similarly
 * for std::calloc, std::aligned_alloc, and std::free, so it is enough to 
 * implement the global functions here. */
HIPSYCL_STDPAR_ALLOC
void* malloc(std::size_t size) {
  return hipsycl::stdpar::unified_shared_memory::malloc(size);
}

HIPSYCL_STDPAR_ALLOC
void* aligned_alloc(std::size_t alignment, std::size_t size) {
  return hipsycl::stdpar::unified_shared_memory::malloc(size, alignment);
}

HIPSYCL_STDPAR_FREE
void free(void* ptr) {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

#endif

#endif
