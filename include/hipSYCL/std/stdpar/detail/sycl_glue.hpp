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


#include <cstdlib>
#include <hipSYCL/sycl/sycl.hpp>
#include <new>


namespace hipsycl::stdpar {

/*
class stdpar_runtime {
public:
  static stdpar_runtime& get() {
    static stdpar_runtime rt;
    return rt;
  }
private:
  stdpar_runtime() {}
  ~stdpar_runtime() {}
};*/

class single_device_dispatch {
public:
  static hipsycl::sycl::queue& get_queue() {
    static thread_local hipsycl::sycl::queue q{hipsycl::sycl::property_list{
        hipsycl::sycl::property::queue::in_order{},
        hipsycl::sycl::property::queue::hipSYCL_coarse_grained_events{}}};
    return q;
  }

};


}

#ifdef __clang__


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
    // Seems some apps really on n==0 still returning a valid pointer
    if(n == 0)
      n = 1;
    
    if(thread_local_storage::get().disabled_stack == 0) {
      // To prevent recursion if runtime initialization happens,
      // which requires memory  allocation again.
      push_disabled();
      void* ptr = nullptr;
      if (alignment != 0) {
        ptr = sycl::aligned_alloc_shared(alignment, n,
                                          single_device_dispatch::get_queue());
      } else {
        ptr = sycl::malloc_shared(n, single_device_dispatch::get_queue());
      }
      get()._is_initialized = true;
      pop_disabled();
      return ptr;

    } else {
      if(alignment != 0)
        return ::aligned_alloc(alignment, n);
      else
        return ::malloc(n);
    }
  }

  static void free(void* ptr) {
    if (thread_local_storage::get().disabled_stack == 0 &&
        get()._is_initialized) {
      // TODO Currently we need to prevent allocations from the SYCL interface
      // from being redirected to shared memory, because objects from the SYCL interface
      // may transfer ownership to the runtime.
      // This can however cause issues when the SYCL interface hands objects to user
      // code which then assumes ownership (assume an std::vector returned from some SYCL
      // function). In this scenario, user code will be handed a regular
      // allocation when it expects a shared one.
      // As a hotfix, we check here whether we are actually dealing with a shared allocation.
      push_disabled();
      if (hipsycl::sycl::get_pointer_type(
              ptr, single_device_dispatch::get_queue().get_context()) ==
          hipsycl::sycl::usm::alloc::unknown) {
        ::free(ptr);
      } else {
        sycl::free(ptr, single_device_dispatch::get_queue());
      }
      pop_disabled();
    } else {
      ::free(ptr);
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

    int disabled_stack = 0;
  private:
    thread_local_storage(){}
  };
};

}

// Causes MallocToUSM pass to change visibility as defined for stdpar
// memory management functions (currently hidden)
#define HIPSYCL_STDPAR_MMGMT_VISIBILITY [[clang::annotate("hipsycl_stdpar_mmgmt_visibility")]]


extern "C" void __hipsycl_stdpar_push_disable_usm() noexcept {
  ::hipsycl::stdpar::unified_shared_memory::push_disabled();
}

extern "C" void __hipsycl_stdpar_pop_disable_usm() noexcept {
  ::hipsycl::stdpar::unified_shared_memory::pop_disabled();
}



// This attribute triggers two main things in the MallocToUSM LLVM pass:
// 1. It causes all calls from disallowed code sections (currently everything
// inside ::hipsycl as well as shared_ptr and weak_ptr based on hipsycl types)
// to be surrounded by push_disable/pop_disable calls to locally disable USM
// memory management
// 2. Internalizes linkage to the current LLVM module to not interfere with
// libraries.
#define HIPSYCL_STDPAR_MEMORY_ALLOCATION \
  HIPSYCL_STDPAR_MMGMT_VISIBILITY \
  [[clang::annotate("hipsycl_stdpar_memory_management")]] __attribute__((noinline))

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new(std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new(std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr)
    throw std::bad_alloc{};
  return ptr;
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new(std::size_t n, const std::nothrow_t&) noexcept  {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new[](std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new[](std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_MEMORY_ALLOCATION
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr, std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr, std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr, std::size_t sz ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr, std::size_t sz ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr, std::size_t sz,
                        std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr, std::size_t sz,
                        std::align_val_t al ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr, const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr, const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete  ( void* ptr, std::align_val_t al,
                        const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_MMGMT_VISIBILITY
void operator delete[]( void* ptr, std::align_val_t al,
                        const std::nothrow_t& tag ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}


#endif

#endif
