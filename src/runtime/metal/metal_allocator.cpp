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

#include "hipSYCL/runtime/metal/metal_allocator.hpp"

#include <Metal/Metal.hpp>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <iostream>

namespace hipsycl {
namespace rt {

namespace {
  uintptr_t align_up(uintptr_t v, size_t align) {
    return (v + align - 1) & ~(uintptr_t)(align - 1);
  }

  size_t get_total_ram() {
    uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0)
      return static_cast<std::size_t>(mem);
    return 8ULL << 30; // fallback: 8 GiB
  }

  // Metal allocates MTL::Buffer storage in page_size units and always appends
  // one extra gap page after the user data.  The total page count must be even,
  // so we round up (sz + 1 gap page) to the next multiple of 2 pages.
  size_t metal_gpu_stride(size_t sz, size_t page_size) {
    const size_t two_pages = 2 * page_size;
    return (sz + page_size + two_pages - 1) & ~(two_pages - 1);
  }
}

static constexpr double mmap_region_size_fraction = 10.0;

struct metal_mmap_region {
  metal_mmap_region(size_t capacity, size_t alignment)
    : _mmap_size(capacity), _capacity(capacity), _alignment(alignment)
  {
    void* ptr = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED) {
      throw std::runtime_error("metal_mmap_region: mmap failed");
    }

    _base = static_cast<char*>(ptr);
    auto current = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(_base), _alignment));
    capacity = capacity - (current - _base);
    _midpoint = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(current + capacity / 2), _alignment));

    madvise(current, _capacity, MADV_FREE_REUSABLE);
    _free_blocks.emplace(current, _capacity);
  }

  ~metal_mmap_region() {
    if (_base) {
      munmap(_base, _mmap_size);
    }
  }

  void* midpoint() {
    return static_cast<void*>(_midpoint);
  }

  void* alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }
    size = align_up(size, _alignment);

    std::lock_guard<std::mutex> lock{_mutex};

    for (auto it = _free_blocks.begin(); it != _free_blocks.end(); ++it) {
      if (it->second >= size) {
        void* ptr = it->first;
        size_t remaining = it->second - size;
        _free_blocks.erase(it);
        if (remaining > 0) {
          _free_blocks.emplace(static_cast<char*>(ptr) + size, remaining);
        }
        madvise(ptr, size, MADV_FREE_REUSE);
        return ptr;
      }
    }

    return nullptr;
  }

  void* alloc_at(void* addr, size_t size) {
    if (addr == nullptr) {
      return alloc(size);
    }
    if (size == 0) {
      return nullptr;
    }
    const size_t requested_size = size;
    size = align_up(size, _alignment);

    std::lock_guard<std::mutex> lock{_mutex};

    uintptr_t aligned_addr = align_up(reinterpret_cast<uintptr_t>(addr), _alignment);
    if (aligned_addr != reinterpret_cast<uintptr_t>(addr)) {
      dump_alloc_at_failure(addr, requested_size, size, "requested address is not aligned");
      return nullptr;
    }

    auto it = _free_blocks.upper_bound(addr);
    if (it == _free_blocks.begin()) {
      dump_alloc_at_failure(addr, requested_size, size, "requested address is before the first free block");
      return nullptr;
    }
    --it;
    char* ptr = static_cast<char*>(it->first);
    if (ptr + it->second <= static_cast<char*>(addr)) {
      // addr is after the end of the free block
      dump_alloc_at_failure(addr, requested_size, size, "requested address is after the containing free block");
      return nullptr;
    }
    if (static_cast<char*>(addr) + size > ptr + it->second) {
      // addr + size exceeds the end of the free block
      dump_alloc_at_failure(addr, requested_size, size, "requested range exceeds the containing free block");
      return nullptr;
    }
    size_t remaining = (ptr + it->second) - (static_cast<char*>(addr) + size);
    _free_blocks.erase(it);
    if (remaining > 0) {
      madvise(static_cast<char*>(addr) + size, remaining, MADV_FREE_REUSABLE);
      _free_blocks.emplace(static_cast<char*>(addr) + size, remaining);
    }
    if (ptr < static_cast<char*>(addr)) {
      madvise(ptr, static_cast<char*>(addr) - ptr, MADV_FREE_REUSABLE);
      _free_blocks.emplace(ptr, static_cast<char*>(addr) - ptr);
    }
    madvise(addr, size, MADV_FREE_REUSE);
    return addr;
  }

  void dump_alloc_at_failure(void* addr, size_t requested_size,
                             size_t aligned_size, const char* reason) const {
    std::cerr << "metal_mmap_region::alloc_at failed: " << reason
              << ", requested=[" << addr << ", "
              << static_cast<void*>(static_cast<char*>(addr) + aligned_size)
              << "), requested_size=" << requested_size
              << ", aligned_size=" << aligned_size << "\n";
    std::cerr << "free blocks:\n";
    for (const auto& [free_ptr, free_size] : _free_blocks) {
      std::cerr << "  [" << free_ptr << ", "
                << static_cast<void*>(static_cast<char*>(free_ptr) + free_size)
                << "), size=" << free_size << "\n";
    }
  }

  void free(void* ptr, size_t size) {
    if (!ptr || size == 0) {
      return;
    }

    void* end = static_cast<char*>(ptr) + size;

    std::lock_guard<std::mutex> lock{_mutex};

    madvise(ptr, size, MADV_FREE_REUSABLE);
    auto [it, _] = _free_blocks.emplace(ptr, size);

    auto next = std::next(it);
    if (next != _free_blocks.end() && next->first == end) {
      it->second += next->second;
      _free_blocks.erase(next);
    }

    if (it != _free_blocks.begin()) {
      auto prev = std::prev(it);
      if (static_cast<char*>(prev->first) + prev->second == it->first) {
        prev->second += it->second;
        _free_blocks.erase(it);
      }
    }
  }

  char* _base;
  char* _midpoint;
  size_t _mmap_size;
  size_t _capacity;
  size_t _alignment;

  std::mutex _mutex;
  std::map<void*, size_t> _free_blocks;
};

metal_allocator::metal_allocator(MTL::Device* device, const device_id &id)
  : _device{device}
  , _device_id{id}
  , _page_size{static_cast<size_t>(getpagesize())}
  , _delta{(size_t)-1}
  , _mmap_region(std::make_shared<metal_mmap_region>(
      static_cast<size_t>(get_total_ram() * mmap_region_size_fraction), _page_size))
{
  calibrate();
}

metal_allocator::~metal_allocator() {
  for (auto& [ptr, block] : _ptr_to_block) {
    if (block.buffer) {
      block.buffer->release();
    }
  }
}

void* metal_allocator::raw_allocate(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto storage_mode = MTL::ResourceStorageModePrivate;
  auto buffer = _device->newBuffer(size_bytes, storage_mode);
  void* gpu_ptr = reinterpret_cast<void*>(buffer->gpuAddress());
  auto block = usm_block{
    .buffer = buffer,
    .alloc_type = usm_alloc_type::device
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[gpu_ptr] = block;
  return gpu_ptr;
}

void *metal_allocator::raw_allocate_usm(
  size_t size_bytes,
  const allocation_hints &hints)
{
  auto buffer = alloc_buffer(size_bytes);
  if (!buffer) {
    return nullptr;
  }
  void* host_ptr = buffer->contents();
  auto block = usm_block{
    .buffer = buffer,
    .alloc_type = usm_alloc_type::shared,
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[host_ptr] = block;
  return host_ptr;
}

void *
metal_allocator::raw_allocate_optimized_host(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto buffer = alloc_buffer(size_bytes);
  if (!buffer) {
    return nullptr;
  }
  void* host_ptr = buffer->contents();
  auto block = usm_block{
    .buffer = buffer,
    .alloc_type = usm_alloc_type::host,
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[host_ptr] = block;
  return host_ptr;
}

void metal_allocator::raw_free(void *mem)
{
  if (!mem) return;

  std::lock_guard<std::mutex> lock{_mutex};
  auto it = _ptr_to_block.find(mem);
  if (it != _ptr_to_block.end()) {
    std::cerr << "metal_allocator::raw_free: mem=" << mem
              << ", buffer=" << it->second.buffer << "\n";
    if(it->second.buffer) {
      it->second.buffer->release();
    } else {
      std::free(mem);
    }
    _ptr_to_block.erase(it);
  }
}

bool metal_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  return b.id == backend_id::metal;
}

result metal_allocator::query_pointer(
  const void *ptr,
  pointer_info &out) const
{
  memset(&out, 0, sizeof(pointer_info));
  out.dev = _device_id;
  if (!ptr) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Null pointer queried"});
  }
  auto [buffer, offset, alloc_type] = get_usm_block(ptr);
  if (alloc_type == usm_alloc_type::undefined) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Pointer is unknown"});
  }
  if (alloc_type == usm_alloc_type::host) {
    out.is_optimized_host = true;
    return make_success();
  }
  if (alloc_type == usm_alloc_type::shared) {
    out.is_usm = true;
    return make_success();
  }

  return make_success();
}

result metal_allocator::mem_advise(
  const void *addr, std::size_t num_bytes,
  int advise) const
{
  return make_success();
}

device_id metal_allocator::get_device() const {
  return _device_id;
}

MTL::Buffer* metal_allocator::alloc_buffer(size_t size_bytes) {
  const size_t aligned = align_up(size_bytes, _page_size);
  const size_t stride  = metal_gpu_stride(aligned, _page_size);
  void* region_ptr = nullptr;
  size_t delta = 0;
  MTL::Buffer* buffer = nullptr;

  auto make_buffer = [this, stride, aligned](MTL::Buffer* buffer, void* region_ptr) -> MTL::Buffer* {
    if (buffer) {
      region_ptr = reinterpret_cast<void*>(buffer->gpuAddress() - _delta);
      buffer->release();
    }
    region_ptr = _mmap_region->alloc_at(region_ptr, stride);
    if (!region_ptr) {
      return nullptr;
    }

    auto mmap_region = _mmap_region;
    buffer = _device->newBuffer(
      region_ptr, aligned, MTL::ResourceStorageModeShared,
      ^(void*, NS::UInteger) {
        std::cerr << "metal_allocator: releasing mmap-backed buffer region=["
                  << region_ptr << ", "
                  << static_cast<void*>(static_cast<char*>(region_ptr) + stride)
                  << "), stride=" << stride << "\n";
        mmap_region->free(region_ptr, stride);
      });
    if (!buffer) {
      mmap_region->free(region_ptr, stride);
      return nullptr;
    }

    return buffer;
  };

  buffer = make_buffer(buffer, region_ptr);
  if (!buffer) {
    return nullptr;
  }
  delta = buffer->gpuAddress() - reinterpret_cast<uintptr_t>(buffer->contents());
  int it = 0, max_iterations = 10;
  while (delta != _delta) {
    buffer = make_buffer(buffer, region_ptr);
    if (!buffer) {
      return nullptr;
    }
    delta = buffer->gpuAddress() - reinterpret_cast<uintptr_t>(buffer->contents());
    if (++it > max_iterations) {
      buffer->release();
      return nullptr;
    }
  }
  return buffer;
}

void metal_allocator::calibrate() {
  auto* buffer = _device->newBuffer(_mmap_region->midpoint(), _page_size, MTL::ResourceStorageModeShared,
    ^(void*, NS::UInteger) { });
  _delta = buffer->gpuAddress() - reinterpret_cast<uintptr_t>(buffer->contents());
  buffer->release();
}

std::tuple<MTL::Buffer*, size_t, metal_allocator::usm_alloc_type> metal_allocator::get_usm_block(const void* ptr) const {
  std::lock_guard<std::mutex> lock{_mutex};
  if (_ptr_to_block.empty()) {
    return {nullptr, 0, usm_alloc_type::undefined};
  }
  auto it = _ptr_to_block.upper_bound(const_cast<void*>(ptr));
  if (it == _ptr_to_block.begin()) {
    return {nullptr, 0, usm_alloc_type::undefined};
  }
  --it;
  const usm_block& block = it->second;
  size_t offset = static_cast<const char*>(ptr) -
          static_cast<const char*>(it->first);
  if (offset < block.buffer->length()) {
    return {block.buffer, offset, block.alloc_type};
  }
  return {nullptr, 0, usm_alloc_type::undefined};
}

} // namespace rt
} // namespace hipsycl
