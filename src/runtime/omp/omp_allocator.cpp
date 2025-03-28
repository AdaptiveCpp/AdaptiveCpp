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
#include <cstdlib>
#include <mutex>

#ifdef LIB_NUMA_AVAILABLE
#include <numa.h>
#include <vector>
#include <unordered_map>
#endif


#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/omp/omp_allocator.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

#ifdef LIB_NUMA_AVAILABLE
namespace {

static std::mutex numa_amap_mutex;
using numa_amap_t = std::unordered_map<void*, size_t>;

numa_amap_t& get_numa_allocation_map() {
  static numa_amap_t numa_amap;
  return numa_amap;
}

}
#endif

omp_allocator::omp_allocator(const device_id &my_device)
    : _my_device{my_device} {}

void *omp_allocator::raw_allocate(size_t min_alignment, size_t size_bytes,
                                  const allocation_hints &hints) {
  if(min_alignment < 32) {
    // Enforce alignment by default for performance reasons.
    // 32 is chosen since this is what is currently needed by the adaptivity
    // engine to consider an allocation strongly aligned.
    return raw_allocate(32, size_bytes, hints);
  }


static bool has_warned = false;
#ifdef LIB_NUMA_AVAILABLE
  //verify that the libnuma can be used on this machine.
  static const bool target_numa_node_available = (numa_available() != -1);
  if(hints.AdaptiveCpp_target_numa_node && target_numa_node_available){
    std::vector<size_t> numa_hints = hints.AdaptiveCpp_target_numa_node.value();
    if(numa_hints.empty()) {
      HIPSYCL_DEBUG_ERROR << "omp_allocator: at least one node must be "
                          << "specified in the AdaptiveCpp_target_numa_node "
                          << "property" << std::endl;
      return nullptr;
    }

    // get the bitmask descibing the available numa nodes
    struct bitmask *available_bm = numa_get_mems_allowed();
    struct bitmask *bm = numa_allocate_nodemask();

    for(size_t node_id : numa_hints){
      // verify that the node requested is available
      if(numa_bitmask_isbitset(available_bm, node_id) == 0){
        HIPSYCL_DEBUG_ERROR << "omp_allocator: the numa node "
                            << "'" << node_id << "' "
                            << "requested in the AdaptiveCpp_target_numa_node "
                            << "property is out of range" << std::endl;
        return nullptr;
      }
      bm = numa_bitmask_setbit(bm, node_id);
    }

    // numa_alloc_interleaved align at PAGESIZE
    // so min_alignment requirement is always met
    void* mem = numa_alloc_interleaved_subset(size_bytes, bm);
    numa_free_nodemask(bm);
    numa_free_nodemask(available_bm);

    // return if allocation fails
    if (!mem)
      return mem;

    // the allocation size is stored in an allocation map so that it can be
    // retrieved when calling numafree.
    numa_amap_t& numa_amap = get_numa_allocation_map();
    std::lock_guard<std::mutex> lock(numa_amap_mutex);
    numa_amap[mem] = size_bytes;

    return mem;
  }
  else if(hints.AdaptiveCpp_target_numa_node && !target_numa_node_available &&
          !has_warned){
      //if the numa library can't be used and the user uses the target_numa_node
      //property, warn him and ignore the property.
      has_warned = true;
      HIPSYCL_DEBUG_WARNING << "omp_allocator: Libnuma cannot be used on this "
        << "machine. Using the target_numa_node property "
        << "will have no effect. "<< std::endl;

    }
#else
  if(hints.AdaptiveCpp_target_numa_node && !has_warned){
      has_warned = true;
      HIPSYCL_DEBUG_WARNING << "omp_allocator: AdaptiveCpp was not built with "
        << "libnuma support. Using the target_numa_node property "
        << "will have no effect. "<< std::endl;
  }
#endif // LIB_NUMA_AVAILABLE

#if !defined(_WIN32)
  // posix requires alignment to be a multiple of sizeof(void*)
  if (min_alignment < sizeof(void*))
    return std::malloc(size_bytes);
#else
  /* The std::free function of the Microsoft C Runtime Library cannot handle
     aligned memory, therefore omp_allocator::free always calls _aligned_free.
     This, however, can only free memory allocated with _aligned_malloc, but
     _aligned_malloc returns NULL when min_alignment == 0.  */
  if (min_alignment == 0)
    min_alignment = 1;
#endif

  if(min_alignment > 0 && size_bytes % min_alignment != 0)
    return raw_allocate(min_alignment,
                        next_multiple_of(size_bytes, min_alignment), hints);

    // ToDo: Mac OS CI has a problem with std::aligned_alloc
    // but it's unclear if it's a Mac, or libc++, or toolchain issue
#ifdef __APPLE__
  return aligned_alloc(min_alignment, size_bytes);
#elif !defined(_WIN32)
  return std::aligned_alloc(min_alignment, size_bytes);
#else
  min_alignment = power_of_2_ceil(min_alignment);
  return _aligned_malloc(size_bytes, min_alignment);
#endif
}

void *omp_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                                 size_t bytes,
                                                 const allocation_hints &hints) {
  return this->raw_allocate(min_alignment, bytes, hints);
};

void omp_allocator::raw_free(void *mem) {
#ifdef LIB_NUMA_AVAILABLE
  {
    std::lock_guard<std::mutex> lock(numa_amap_mutex);
    numa_amap_t& numa_amap = get_numa_allocation_map();
    if(auto node = numa_amap.extract(mem)){
      size_t size_bytes = node.mapped();
      numa_free(mem, size_bytes);
      return;
    }
  }
#endif // LIB_NUMA_AVAILABLE

#if !defined(_WIN32)
  std::free(mem);
#else
  _aligned_free(mem);
#endif
}

void* omp_allocator::raw_allocate_usm(size_t bytes,
                                      const allocation_hints &hints) {
  return this->raw_allocate(0, bytes, hints);
}

bool omp_allocator::is_usm_accessible_from(backend_descriptor b) const {
  if(b.hw_platform == hardware_platform::cpu) {
    return true;
  }
  return false;
}

device_id omp_allocator::get_device() const {
  return _my_device;
}

result omp_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  
  // For a host device, USM is the same as host memory?
  out.is_optimized_host = true;
  out.is_usm = true;
  out.is_from_host_backend = true;
  out.dev = _my_device;

  return make_success();
}

result omp_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                 int advise) const {
  HIPSYCL_DEBUG_WARNING << "omp_allocator: Ignoring mem_advise() hint"
                        << std::endl;
  return make_success();
}

}
}
