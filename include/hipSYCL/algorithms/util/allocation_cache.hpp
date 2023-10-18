
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


#ifndef HIPSYCL_ALGORITHM_UTIL_ALLOCATION_CACHE_HPP
#define HIPSYCL_ALGORITHM_UTIL_ALLOCATION_CACHE_HPP

#include <vector>
#include <mutex>

#include "hipSYCL/common/small_vector.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/sycl/device.hpp"

namespace hipsycl::rt {
  class runtime;
}

namespace hipsycl::algorithms::util {


struct allocation {
  void* ptr;
  std::size_t size;
  rt::device_id dev;
};

class allocation_group;

enum class allocation_type {
  device, shared, host
};

class allocation_cache {
  friend class allocation_group;
public:
  allocation_cache(allocation_type alloc_type)
  : _alloc_type{alloc_type} {}

  ~allocation_cache() {
    purge();
  }

  void purge() {
    std::lock_guard<std::mutex> lock{_mutex};
    
    for(auto& allocation : _allocations) {
      _rt.get()->backends()
          .get(allocation.dev.get_backend())
          ->get_allocator(allocation.dev)
          ->free(allocation.ptr);
    }
    _allocations.clear();
  }
private:
  
  allocation find_or_alloc(std::size_t min_size, std::size_t min_alignment,
                           rt::device_id dev) {
    allocation result;
    if(!find_allocation(min_size, min_alignment, dev, result)){  
      result.dev = dev;
      result.size = min_size;

      auto allocator = _rt.get()->backends()
                       .get(dev.get_backend())
                       ->get_allocator(dev);

      if(_alloc_type == allocation_type::device)
        result.ptr = allocator->allocate(min_alignment, min_size);
      else if(_alloc_type == allocation_type::shared)
        result.ptr = allocator->allocate_usm(min_size);
      else
        result.ptr =
            allocator->allocate_optimized_host(min_alignment, min_size);
    }
    return result;
  }

  bool find_allocation(std::size_t min_size, std::size_t min_alignment,
                       rt::device_id dev, allocation &out) {
    std::lock_guard<std::mutex> lock{_mutex};

    bool found = false;
    std::size_t found_index = 0;
    for (std::size_t i = 0; i < _allocations.size(); ++i) {
      const auto& allocation = _allocations[i];
      if (allocation.dev == dev) {
        if (allocation.size >= min_size &&
            reinterpret_cast<std::size_t>(allocation.ptr) % min_alignment ==
                0) {
          // If we already have found a candidate: We want the smallest
          // allocation that has the required size so that larger allocations
          // remain available for larger requests.
          if (!found || allocation.size < out.size) {
            out = allocation;
            found_index = i;
          }
          found = true;
        }
      }
    }
    if(found) {
      // The allocation is no longer available for other requests,
      // so remove for now.
      _allocations.erase(_allocations.begin() + found_index);
    }
    return found;
  }

  void return_allocation(const allocation& alloc) {
    std::lock_guard<std::mutex> lock{_mutex};
    _allocations.push_back(alloc);
  }

  rt::runtime_keep_alive_token _rt;
  common::auto_small_vector<allocation> _allocations;
  std::mutex _mutex;
  allocation_type _alloc_type;
};

/// allocation_group represents allocation requests that belong together
/// semantically and share lifetime - for example, all scratch allocations
/// required by one reduction invocation.
///
/// allocation_group is not thread-safe.
///
/// When the object releases its managed allocations, they are returned to the
/// parent cache from where they might be reused to serve other allocation
/// requests.
/// The user should therefore ensure that the lifetime of this object is either
/// at least as long as the operations using its allocations,
/// or all operations are ordered such that there is no hazard of race conditions, e.g.
/// if only in-order queues are involved and one allocation_cache exists per in-order queue.
class allocation_group {
public:
  allocation_group(allocation_cache *parent_cache, rt::device_id dev)
      : _parent{parent_cache}, _dev{dev} {}

  allocation_group(allocation_cache *parent_cache, const sycl::device &dev)
      : _parent{parent_cache},
        _dev{dev.hipSYCL_device_id()} {}

  allocation_group() = default;
  allocation_group(const allocation_group&) = delete;
  allocation_group& operator=(const allocation_group&) = delete;

  ~allocation_group() {
    release();
  }

  void release() {
    for(const auto& allocation : _managed_allocations) {
      _parent->return_allocation(allocation);
    }
    _managed_allocations.clear();
  }

  template<class T>
  T* obtain(std::size_t count) {
    allocation alloc =
        _parent->find_or_alloc(count * sizeof(T), alignof(T), _dev);
    _managed_allocations.push_back(alloc);
    return static_cast<T*>(alloc.ptr);
  }

  rt::device_id get_device() const {
    return _dev;
  }
private:
  allocation_cache* _parent;
  rt::device_id _dev;
  common::auto_small_vector<allocation> _managed_allocations;
};


}

#endif
