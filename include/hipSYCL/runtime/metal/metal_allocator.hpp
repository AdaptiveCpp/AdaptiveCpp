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

#include <map>
#include <memory>

namespace MTL {

class Device;
class Buffer;
class ResidencySet;

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

  // Null if the device does not provide residency sets
  MTL::ResidencySet* get_residency_set() const { return _residency_set; }

  template<typename F>
  void for_each_buffer(F&& f) const {
    std::lock_guard<std::mutex> lock{_mutex};
    for (auto& [ptr, block] : _ptr_to_block) {
      if (block.buffer) {
        f(block.buffer);
      }
    }
  }

private:
  MTL::Buffer* alloc_buffer(size_t size_bytes);
  void calibrate();
  void add_to_residency_set(MTL::Buffer* buffer);
  void remove_from_residency_set(MTL::Buffer* buffer);

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
  MTL::ResidencySet* _residency_set = nullptr;
  std::shared_ptr<metal_mmap_region> _mmap_region;
};



} // namespace rt
} // namespace hipsycl

#endif
