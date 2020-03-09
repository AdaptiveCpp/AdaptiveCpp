/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#include "hipSYCL/sycl/accessor.hpp"
#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/detail/buffer.hpp"
#include "hipSYCL/sycl/detail/task_graph.hpp"
#include "hipSYCL/sycl/detail/application.hpp"
#include "hipSYCL/sycl/detail/debug.hpp"

#include <cassert>

namespace hipsycl {
namespace sycl {
namespace detail {
namespace accessor {

void* obtain_host_access(buffer_ptr buff,
                         access::mode access_mode)
{

  void* ptr = buff->get_host_ptr();
  stream_ptr stream = stream_manager::default_stream();

  HIPSYCL_DEBUG_INFO << "accessor: Spawning host access task"
                     << std::endl;

  auto task_graph_node = detail::buffer_impl::access_host(
        buff,
        access_mode,
        stream,
        stream->get_error_handler());

  task_graph_node->wait();

  return ptr;

}


void* obtain_device_access(buffer_ptr buff,
                           sycl::handler& cgh,
                           access::mode access_mode)
{
  void* access_ptr = nullptr;

#ifndef HIPSYCL_CPU_EMULATE_SEPARATE_MEMORY
  if(cgh.get_stream()->get_device().is_host())
  {
    // The "device" access that we make is actually a "host" access
    // since we are running on the host device!
    // This happens when we construct an accessor for a kernel (which by definition runs
    // on "device") for a host device.
    //
    // Treat this as host access to avoid unecessary data copies.
    // Note that in this case, it may actually hold that 
    // buff->get_host_ptr() == buff->get_buffer_ptr() if we run on pure CPU,
    // since then a separate device data buffer is not needed.
    access_ptr = buff->get_host_ptr();
    
    auto task_graph_node =
        detail::buffer_impl::access_host(buff,
                                         access_mode,
                                         cgh.get_stream(),
                                         cgh.get_stream()->get_error_handler());

    cgh.add_access(buff, access_mode, task_graph_node);
  }
  else
#endif
  {
    access_ptr = buff->get_buffer_ptr();

    auto task_graph_node =
        detail::buffer_impl::access_device(buff,
                                            access_mode,
                                            cgh.get_stream(),
                                            cgh.get_stream()->get_error_handler());

    cgh.add_access(buff, access_mode, task_graph_node);

  }
  return access_ptr;
}

accessor_id request_accessor_id(buffer_ptr buff, sycl::handler& cgh)
{
  return cgh.request_accessor_id(buff);
}

}
}
}
}
