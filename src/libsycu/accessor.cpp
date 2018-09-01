/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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

#include "CL/sycl/accessor.hpp"
#include "CL/sycl/handler.hpp"
#include "CL/sycl/detail/buffer.hpp"
#include "CL/sycl/detail/task_graph.hpp"
#include "CL/sycl/detail/application.hpp"

#include <cassert>
#include <iostream>

namespace cl {
namespace sycl {
namespace detail {
namespace accessor {

void* obtain_host_access(buffer_ptr buff,
                         access::mode access_mode)
{

  void* ptr = buff->get_host_ptr();
  stream_ptr stream = stream_manager::default_stream();

  std::cout << "Accessing host" << std::endl;
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
  void* ptr = buff->get_buffer_ptr();

  auto task_graph_node =
      detail::buffer_impl::access_device(buff,
                                         access_mode,
                                         cgh.get_stream(),
                                         cgh.get_stream()->get_error_handler());

  cgh._detail_add_access(buff, access_mode, task_graph_node);

  return ptr;
}


}
}
}
}
