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

#include "hipSYCL/runtime/hip/hip_shared_usm_emu.hpp"
#include "hipSYCL/runtime/hip/hip_allocator.hpp"

#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>

namespace hipsycl {
namespace rt {

namespace {

void make_allocation_unavailable(const void* ptr, std::size_t bytes) {
  madvise(const_cast<void*>(ptr), bytes, MADV_DONTNEED);
}

}

hip_shared_usm_emulation::hip_shared_usm_emulation(hip_allocator* alloc)
: _alloc{alloc} {
  _uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK);

  struct uffdio_api ua = {
    .api = UFFD_API
  };

  ioctl(_uffd, UFFDIO_API, &ua);

  if (ua.api != UFFD_API) {
    //TODO: mismatch: kernel doesn't support this API version
  }
}

void hip_shared_usm_emulation::register_queue_client(hip_queue* q){
  assert(q);
  _all_queues.push_back(q);
}

void hip_shared_usm_emulation::unregister_queue_client(hip_queue* q){
  _all_queues.erase(std::remove(_all_queues.begin(), _all_queues.end(), q),
                    _all_queues.end());
  _queues_using_device_alloc.erase(
      std::remove(_queues_using_device_alloc.begin(),
                  _queues_using_device_alloc.end(), q),
      _queues_using_device_alloc.end());
}

void* hip_shared_usm_emulation::malloc(std::size_t bytes){

  void *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  memset(ptr, 0, bytes);

  struct uffdio_register reg = {
    .range = {
        .start = (unsigned long)ptr,
        .len = bytes,
    },
    .mode = UFFDIO_REGISTER_MODE_MISSING,
  };
  ioctl(_uffd, UFFDIO_REGISTER, &reg);

  if(!ptr)
    return ptr;
  
  allocation alloc;
  alloc.base = ptr;
  alloc.device_shadow = nullptr;
  alloc.size = size;

  _allocs.push_back(alloc);

  return alloc.base;
}

void hip_shared_usm_emulation::free(const void* ptr) {
  auto it = std::find(_allocs.begin, _allocs.end(), [=](auto& elem){
    return elem.base == ptr;
  });

  if(it != _allocs.end()) {
    munmap(it->base, it->size);
    if(it->device_shadow)
      deallocate(_alloc, it->device_shadow);
    
    _allocs.erase(it);
  }
}

bool hip_shared_usm_emulation::is_managed_memory(const void* ptr) const {
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  for(auto& alloc : _allocs) {
    intptr_t base_begin = reinterpret_cast<intptr_t>(alloc.base);
    if(intptr >= base_begin && intptr < base_begin + alloc.size)
      return true;
  }
  return false;
}

void hip_shared_usm_emulation::handle_kernel_arguments(
    hip_queue *q, void **mapped_kernel_args, std::size_t num_args,
    const hcf_kernel_info *info, const code_object *obj,
    const std::optional<std::vector<int>> &dae_retained_arguments_mask) {

  if(obj->get_jit_output_metadata().is_free_of_indirect_access) {

    static std::vector<allocation*> allocs_to_migrate;
    allocs_to_migrate.clear();

    for(std::size_t i = 0; i < num_args; ++i) {
      std::size_t original_index = i;
      if(dae_retained_arguments_mask.has_value()) {
        original_index = dae_retained_arguments_mask.value()[i];
      }

      if(info->get_argument_type(original_index) ==
            hcf_kernel_info::argument_type::pointer) {

        const void* arg_location = mapped_kernel_args[i];
        void* ptr;
        std::memcpy(&ptr, arg_location, sizeof(void*));

        allocs_to_migrate.push_back(ptr);  
      }
    }

  } else {
    // TODO
  }
}

void hip_shared_usm_emulation::handle_kernel_submitted() {

}

}
}