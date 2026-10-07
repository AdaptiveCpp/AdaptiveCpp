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

#include <memory>

namespace hipsycl::rt {

std::shared_ptr<allocation_map_t> get_allocation_map() {
  static std::shared_ptr<allocation_map_t> amap = std::make_shared<allocation_map_t>();
  return amap;
}

bool allocation_tracker::register_allocation(const void *ptr, std::size_t size,
                         const allocation_info &info) {
  using value_type = allocation_map_t::value_type;

  value_type v;
  v.allocation_info::operator=(info);
  v.allocation_size = size;
  return get_allocation_map()->insert(reinterpret_cast<uint64_t>(ptr), v);
}

bool allocation_tracker::unregister_allocation(const void* ptr) {
  return get_allocation_map()->erase(reinterpret_cast<uint64_t>(ptr));
}

bool allocation_tracker::query_allocation(const void *ptr, allocation_info &out,
                                          uint64_t &root_address) {
  return get_allocation_map()->get_entry(reinterpret_cast<uint64_t>(ptr), root_address);
}
}
