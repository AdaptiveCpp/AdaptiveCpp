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
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/runtime.hpp"

namespace hipsycl::rt::pcuda {

namespace {

std::vector<pcuda::stream*> stream_registry;
std::mutex stream_registry_lock;

}

pcudaError_t stream::create(pcuda::stream *&out, pcuda_runtime *pcuda_rt,
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
  out = new pcuda::stream{};
  out->_executor = std::shared_ptr<inorder_executor>{exec};

  {
    std::lock_guard<std::mutex> lock{stream_registry_lock};
    stream_registry.push_back(out);
  }

  return pcudaSuccess;
}

pcudaError_t stream::destroy(stream *stream, pcuda_runtime *) {

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

inorder_queue* stream::get_queue() const {
  return _executor.get()->get_queue();
}

inorder_queue* stream::get_queue(pcudaStream_t s) {
  return static_cast<pcuda::stream*>(s)->get_queue();
}

std::shared_ptr<inorder_executor> stream::get_executor() const {
  return _executor;
}

pcudaError_t stream::wait_all(rt::device_id dev) {
  std::vector<pcuda::stream> streams_to_wait;
  {
    std::lock_guard<std::mutex> lock{stream_registry_lock};
    for(int i = 0; i < stream_registry.size(); ++i) {
      if(stream_registry[i]->get_queue()->get_device() == dev) {
        streams_to_wait.push_back(*stream_registry[i]);
      }
    }
  }
  for(auto& s : streams_to_wait) {
    s.get_queue()->wait();
  }
  return pcudaSuccess;
}

}

