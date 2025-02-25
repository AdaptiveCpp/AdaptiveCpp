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

#include <cassert>
#include <mutex>


#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/runtime.hpp"

namespace hipsycl::rt::pcuda {

namespace {

std::vector<internal_stream_t*> stream_registry;
std::mutex stream_registry_lock;

}

pcudaError_t stream_create(internal_stream_t *&out, pcuda_runtime *pcuda_rt,
                           device_id dev, unsigned int flags, int priority){
  assert(pcuda_rt);
  auto executor = pcuda_rt->get_rt()
                      ->backends()
                      .get(dev.get_backend())
                      ->create_inorder_executor(dev, priority);

  if(!executor) {
    register_pcuda_error(__acpp_here(), pcudaErrorUnknown,
                         "Could not construct backend inorder queue");
    return pcudaErrorUnknown;
  }

  inorder_executor* exec = static_cast<inorder_executor*>(executor.release());
  out = new std::shared_ptr<inorder_executor>{exec};

  {
    std::lock_guard<std::mutex> lock{stream_registry_lock};
    stream_registry.push_back(out);
  }

  return pcudaSuccess;
}

pcudaError_t stream_destroy(internal_stream_t *stream, pcuda_runtime *) {

  if(!stream)
    return pcudaSuccess;

  {
    std::lock_guard<std::mutex> lock{stream_registry_lock};
    for(int i = 0; i < stream_registry.size(); ++i) {
      if(stream_registry[i] == stream) {
        stream_registry.erase(stream_registry.begin()+i);
        break;
      }
    }
  }

  
  delete stream;
  return pcudaSuccess;
}

inorder_queue* stream_get(pcudaStream_t stream) {
  return static_cast<internal_stream_t*>(stream)->get()->get_queue();
}

pcudaError_t stream_wait_all(rt::device_id dev) {
  std::vector<internal_stream_t> streams_to_wait;
  {
    std::lock_guard<std::mutex> lock{stream_registry_lock};
    for(int i = 0; i < stream_registry.size(); ++i) {
      if(stream_get(stream_registry[i])->get_device() == dev) {
        streams_to_wait.push_back(*stream_registry[i]);
      }
    }
  }
  for(auto& s : streams_to_wait) {
    s->wait();
  }
  return pcudaSuccess;
}

}

