/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/hw_model/hw_model.hpp"
#include <memory>
#include <mutex>

namespace hipsycl {
namespace rt {

namespace {

class rt_manager
{
public:

  void shutdown() {
    std::lock_guard<std::mutex> lock{_lock};
    rt.reset();
  }

  void reset() {
    HIPSYCL_DEBUG_INFO << "rt_manager: Restarting runtime..." << std::endl;
    
    std::lock_guard<std::mutex> lock{_lock};
    rt.reset();
    // TODO: Reset devices?
  }

  runtime *get_runtime() {
    std::lock_guard<std::mutex> lock{_lock};

    if(!rt)
      rt = std::make_unique<runtime>();
    return rt.get();
  }

  static rt_manager& get() {
    static rt_manager mgr;
    return mgr;
  }

private:
  rt_manager() {}

  std::unique_ptr<runtime> rt;
  mutable std::mutex _lock;
};

}

runtime& application::get_runtime(){
  return *rt_manager::get().get_runtime();
}

dag_manager &application::dag()
{ return get_runtime().dag(); }

backend &application::get_backend(hipsycl::rt::backend_id id)
{
  return *(get_runtime().backends().get(id));
}

void application::reset() {
  rt_manager::get().reset();
}



}
}

