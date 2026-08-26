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
#ifndef HIPSYCL_METAL_ALLOCATOR_HPP
#define HIPSYCL_METAL_ALLOCATOR_HPP

#include "../allocator.hpp"
#include "../hints.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <vector>

namespace MTL {

class Device;
class Buffer;
class Resource;

} // namespace MTL

namespace hipsycl {
namespace rt {

struct metal_mmap_region;

// Metal USM allocator. Reserves a large VA range via mmap and wraps sub-regions
// with newBuffer(ptr,...) so that all shared/host buffers satisfy:
//   buffer->gpuAddress() - (uintptr_t)buffer->contents() == _delta (constant).
// _delta is used for pointer translation between CPU and GPU address spaces.
class metal_allocator : public backend_allocator
{
public:
  enum class usm_alloc_type {
    shared = 0,
    device = 1,
    host = 2,
    undefined = 3
  };

  metal_allocator(MTL::Device* device, const device_id &id);
  ~metal_allocator();

  virtual void* raw_allocate(size_t min_alignment, size_t size_bytes,
                             const allocation_hints &hints = {}) override;

  virtual void *
  raw_allocate_optimized_host(size_t min_alignment, size_t bytes,
                              const allocation_hints &hints = {}) override;

  virtual void raw_free(void *mem) override;

  virtual void *raw_allocate_usm(size_t bytes,
                                 const allocation_hints &hints = {}) override;
  virtual bool is_usm_accessible_from(backend_descriptor b) const override;

  virtual result query_pointer(const void *ptr,
                               pointer_info &out) const override;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const override;

  virtual device_id get_device() const override;

  // Returns the Metal buffer and offset for a given USM pointer
  std::tuple<MTL::Buffer*, size_t, usm_alloc_type> get_usm_block(const void* ptr) const;

  size_t get_delta() const { return _delta; }

  // Monotonic counter bumped on every allocation and free. Allows consumers
  // (e.g. the queue's residency tracking) to detect allocation changes without
  // taking the allocator lock.
  uint64_t generation() const {
    return _generation.load(std::memory_order_relaxed);
  }

  // Replaces the contents of `out` with all Metal buffers currently backing
  // USM allocations, and returns the generation the snapshot corresponds to.
  // Snapshot and generation are taken under the same lock, so the returned
  // pair is always consistent.
  //
  // Each returned buffer is retain()'d before being handed out: the snapshot
  // may be cached and used by the caller (e.g. passed to useResources())
  // after this lock has been released, and without an extra reference a
  // concurrent raw_free() on another thread could release/deallocate the
  // buffer in that window. The caller owns the extra reference and must
  // release() every entry once it stops using the snapshot (e.g. when
  // replacing it with a newer one, or on destruction).
  // (Not a template: retain() requires MTL::Buffer to be a complete type,
  // which this header intentionally does not pull in - see metal_allocator.cpp.)
  uint64_t snapshot_buffers(std::vector<const MTL::Resource*>& out) const;

private:
  MTL::Buffer* alloc_buffer(size_t size_bytes);
  void calibrate();

  MTL::Device* _device = nullptr;
  device_id _device_id;
  const size_t _page_size;
  size_t _delta;

  struct usm_block {
    MTL::Buffer* buffer;
    usm_alloc_type alloc_type;
  };
  std::map<void*, usm_block> _ptr_to_block;
  mutable std::mutex _mutex;
  std::atomic<uint64_t> _generation{0};
  std::shared_ptr<metal_mmap_region> _mmap_region;
};



} // namespace rt
} // namespace hipsycl

#endif
