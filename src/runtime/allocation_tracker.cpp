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

#include "hipSYCL/runtime/allocation_tracker.hpp"

#include <atomic>

namespace hipsycl::rt {

namespace {

using amap_t = common::allocation_map<allocation_info>;

struct allocation_map_wrapper {
  amap_t amap;

  ~allocation_map_wrapper() {
    is_shut_down.store(true, std::memory_order_release);
  }

  static std::atomic<bool> is_shut_down;
};

std::atomic<bool> allocation_map_wrapper::is_shut_down{false};

amap_t* get_allocation_map() {
  if (allocation_map_wrapper::is_shut_down.load(std::memory_order_acquire))
    return nullptr;
  static allocation_map_wrapper wrapper;
  if (allocation_map_wrapper::is_shut_down.load(std::memory_order_acquire))
    return nullptr;
  return &wrapper.amap;
}

}

void allocation_tracker::init() {
  (void)get_allocation_map();
}

bool allocation_tracker::register_allocation(const void *ptr, std::size_t size,
                         const allocation_info &info) {
  auto* amap = get_allocation_map();
  if (!amap)
    return false;

  using value_type = amap_t::value_type;

  value_type v;
  v.allocation_info::operator=(info);
  v.allocation_size = size;
  return amap->insert(reinterpret_cast<uint64_t>(ptr), v);
}

bool allocation_tracker::unregister_allocation(const void* ptr) {
  auto* amap = get_allocation_map();
  if (!amap)
    return false;

  return amap->erase(reinterpret_cast<uint64_t>(ptr));
}

bool allocation_tracker::query_allocation(const void *ptr, allocation_info &out,
                                          uint64_t &root_address) {
  auto* amap = get_allocation_map();
  if (!amap)
    return false;

  return amap->get_entry(reinterpret_cast<uint64_t>(ptr), root_address);
}
}
