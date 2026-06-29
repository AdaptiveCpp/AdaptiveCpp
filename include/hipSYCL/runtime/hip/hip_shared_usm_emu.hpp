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

#ifndef ACPP_SHARED_USM_EMULATION_HPP
#define ACPP_SHARED_USM_EMULATION_HPP

#include <optional>
#include <vector>


#include "hip_queue.hpp"

namespace hipsycl {
namespace rt {

class hcf_kernel_info;
class code_object;
class hip_allocator;

class hip_shared_usm_emulation {
public:
  hip_shared_usm_emulation(hip_allocator* alloc);

  // All queues running operations on the managed memory
  // must register themselves by calling this function.
  void register_queue_client(hip_queue* q);
  // At queue destruction, queues must call this.
  void unregister_queue_client(hip_queue* q);

  void* malloc(std::size_t bytes);
  void free(const void* ptr);

  void handle_kernel_arguments(
      hip_queue *q, void **mapped_kernel_args, std::size_t num_args,
      const hcf_kernel_info *info, const code_object *obj,
      const std::optional<std::vector<int>> &dae_retained_arguments_mask);

  void handle_kernel_submitted();

  bool is_managed_memory(const void* ptr) const;

private:
  struct allocation {
    const void* base;
    const void* device_shadow;
    std::size_t size;
  };

  allocation* find_allocation(const void* ptr) const;

  hip_allocator* _alloc;
  std::vector<hip_queue*> _queues_using_device_alloc;
  std::vector<hip_queue*> _all_queues;
  // TODO might want to use allocation_map here?
  std::vector<allocation> _allocs;
  int _uffd;

  bool _is_in_fallback_mode = false;
};

}
}

#endif
