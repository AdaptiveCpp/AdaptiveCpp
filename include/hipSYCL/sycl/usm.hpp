/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_USM_HPP
#define HIPSYCL_USM_HPP

#include <cstdint>
#include <cassert>
#include <exception>

#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"
#include "exception.hpp"
#include "usm_query.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/allocator.hpp"

namespace hipsycl {
namespace sycl {


// Explicit USM

inline void *malloc_device(size_t num_bytes, const device &dev,
                           const context &ctx) {
  return detail::select_device_allocator(dev)->allocate(0, num_bytes);
}

template <typename T>
T* malloc_device(std::size_t count, const device &dev,
                       const context &ctx) {
  return static_cast<T*>(malloc_device(count * sizeof(T), dev, ctx));
}

inline void *malloc_device(size_t num_bytes, const queue &q) {
  return malloc_device(num_bytes, q.get_device(), q.get_context());
}

template <typename T>
T* malloc_device(std::size_t count, const queue &q) {
  return malloc_device<T>(count, q.get_device(), q.get_context());
}

inline void *aligned_alloc_device(std::size_t alignment, std::size_t num_bytes,
                                  const device &dev, const context &ctx) {
  return detail::select_device_allocator(dev)->allocate(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const device &dev, const context &ctx) {
  return static_cast<T *>(
      aligned_alloc_device(alignment, count * sizeof(T), dev, ctx));
}

inline void *aligned_alloc_device(std::size_t alignment, std::size_t size,
                                  const queue &q) {
  return aligned_alloc_device(alignment, size, q.get_device(), q.get_context());
}

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const queue &q) {
  return aligned_alloc_device<T>(alignment, count, q.get_device(), q.get_context());
}

// Restricted USM

inline void *malloc_host(std::size_t num_bytes, const context &ctx) {
  return detail::select_usm_allocator(ctx)->allocate_optimized_host(0, num_bytes);
}

template <typename T> T *malloc_host(std::size_t count, const context &ctx) {
  return static_cast<T*>(malloc_host(count * sizeof(T), ctx));
}

inline void *malloc_host(std::size_t num_bytes, const queue &q) {
  return malloc_host(num_bytes, q.get_context());
}

template <typename T> T *malloc_host(std::size_t count, const queue &q) {
  return malloc_host<T>(count, q.get_context());
}

inline void *malloc_shared(std::size_t num_bytes, const device &dev,
                           const context &ctx) {
  return detail::select_usm_allocator(ctx, dev)->allocate_usm(num_bytes);
}

template <typename T>
T *malloc_shared(std::size_t count, const device &dev, const context &ctx) {
  return static_cast<T*>(malloc_shared(count * sizeof(T), dev, ctx));
}

inline void *malloc_shared(std::size_t num_bytes, const queue &q) {
  return malloc_shared(num_bytes, q.get_device(), q.get_context());
}

template <typename T> T *malloc_shared(std::size_t count, const queue &q) {
  return malloc_shared<T>(count, q.get_device(), q.get_context());
}

inline void *aligned_alloc_host(std::size_t alignment, std::size_t num_bytes,
                                const context &ctx) {
  return detail::select_usm_allocator(ctx)->allocate_optimized_host(alignment,
                                                                    num_bytes);
}

template <typename T>
T *aligned_alloc_host(std::size_t alignment, size_t count, const context &ctx) {
  return static_cast<T*>(aligned_alloc_host(alignment, count * sizeof(T), ctx));
}

inline void *aligned_alloc_host(size_t alignment, size_t num_bytes,
                                const queue &q) {
  return aligned_alloc_host(alignment, num_bytes, q.get_context());
}

template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                         const queue &q) {
  return static_cast<T *>(
      aligned_alloc_host(alignment, count * sizeof(T), q.get_context()));
}

inline void *aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes,
                                  const device &dev, const context &ctx) {
  return detail::select_usm_allocator(ctx, dev)->allocate_usm(num_bytes);
}

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const device &dev, const context &ctx) {
  return static_cast<T*>(aligned_alloc_shared(alignment, count * sizeof(T), dev, ctx));
}

inline void *aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes,
                                  const queue &q) {
  return aligned_alloc_shared(alignment, num_bytes, q.get_device(), q.get_context());
}

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const queue &q) {
  return static_cast<T *>(aligned_alloc_shared(
      alignment, count * sizeof(T), q.get_device(), q.get_context()));
}


// General

inline void *malloc(std::size_t num_bytes, const device &dev,
                    const context &ctx, usm::alloc kind) {

  if (kind == usm::alloc::device) {
    return malloc_device(num_bytes, dev, ctx);
  } else if (kind == usm::alloc::host) {
    return malloc_host(num_bytes, ctx);
  } else if (kind == usm::alloc::shared) {
    return malloc_shared(num_bytes, dev, ctx);
  }
  return nullptr;
}

template <typename T>
T *malloc(std::size_t count, const device &dev, const context &ctx,
          usm::alloc kind) {
  return static_cast<T*>(malloc(count * sizeof(T), dev, ctx, kind));
}

inline void *malloc(std::size_t num_bytes, const queue &q, usm::alloc kind) {
  return malloc(num_bytes, q.get_device(), q.get_context(), kind);
}

template <typename T>
T *malloc(std::size_t count, const queue &q, usm::alloc kind) {
  return static_cast<T *>(
      malloc(count * sizeof(T), q.get_device(), q.get_context(), kind));
}

inline void *aligned_alloc(std::size_t alignment, std::size_t num_bytes,
                           const device &dev, const context &ctx,
                           usm::alloc kind) {
  if (kind == usm::alloc::device) {
    return aligned_alloc_device(alignment, num_bytes, dev, ctx);
  } else if (kind == usm::alloc::host) {
    return aligned_alloc_host(alignment, num_bytes, ctx);
  } else if (kind == usm::alloc::shared) {
    return aligned_alloc_shared(alignment, num_bytes, dev, ctx);
  }
  return nullptr;
}

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count, const device &dev,
                 const context &ctx, usm::alloc kind) {
  return static_cast<T *>(
      aligned_alloc(alignment, count * sizeof(T), dev, ctx, kind));
}

inline void *aligned_alloc(std::size_t alignment, std::size_t num_bytes,
                           const sycl::queue &q, usm::alloc kind) {
  return aligned_alloc(alignment, num_bytes, q.get_device(), q.get_context(),
                       kind);
}

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count, const sycl::queue &q,
                 usm::alloc kind) {
  return static_cast<T *>(aligned_alloc(alignment, count * sizeof(T),
                                        q.get_device(), q.get_context(), kind));
}

inline void free(void *ptr, const sycl::context &ctx) {
  return detail::select_usm_allocator(ctx)->free(ptr);
}

inline void free(void *ptr, const sycl::queue &q) {
  free(ptr, q.get_context());
}

// hipSYCL synchronous mem_advise extension
inline void mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const context &ctx, const device &dev) {

  rt::backend_allocator* b = detail::select_usm_allocator(ctx, dev);
  assert(b);

  rt::result r = b->mem_advise(ptr,  num_bytes, advise);

  if(!r.is_success())
    std::rethrow_exception(glue::throw_result(r));
}

inline void mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const queue& q) {
  mem_advise(ptr, num_bytes, advise, q.get_context(), q.get_device());
}

// USM allocator
template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  static_assert(
      AllocKind != usm::alloc::device,
      "usm_allocator does not support AllocKind == usm::alloc::device");

  usm_allocator() noexcept = delete;
  usm_allocator(const context &ctx, const device &dev) noexcept
      : _ctx{ctx}, _dev{dev} {}


  usm_allocator(const queue &q) noexcept
      : _ctx{q.get_context()}, _dev{q.get_device()} {}
  
  usm_allocator(const usm_allocator &) noexcept = default;
  usm_allocator(usm_allocator &&) noexcept = default;

  usm_allocator &operator=(const usm_allocator &) = delete;
  usm_allocator &operator=(usm_allocator &&) = default;

  template <class U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
      : _ctx{other._ctx}, _dev{other._dev} {}

  T *allocate(std::size_t num_elements) {

    T *ptr = aligned_alloc<T>(Alignment, num_elements, _dev, _ctx, AllocKind);

    if (!ptr)
      throw memory_allocation_error("usm_allocator: Allocation failed");

    return ptr;
  }

  void deallocate(T *ptr, std::size_t size) {
    if (ptr)
      free(ptr, _ctx);
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &a,
                         const usm_allocator<U, AllocKindU, AlignmentU> &b) {
    return a._dev == b._dev && a._ctx == b._ctx && AllocKindU == AllocKind &&
           AlignmentU == Alignment;
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &a,
                         const usm_allocator<U, AllocKindU, AlignmentU> &b) {
    return !(a == b);
  }

private:
  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend class usm_allocator;
  context _ctx;
  device _dev;
};
}
} // namespace hipsycl

#endif
