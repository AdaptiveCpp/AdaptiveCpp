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
#include <atomic>

namespace hipsycl {
namespace rt {

namespace {

class rt_manager
{
public:
  void shutdown() {
    // TODO Thread safety...
    delete _rt;
    _rt.store(nullptr);
  }

  void reset() {
    HIPSYCL_DEBUG_INFO << "rt_manager: Restarting runtime..." << std::endl;

    // TODO: This implementation has a curious side effect:
    // When a reset of the runtime is triggered,
    // operations still being processed will already run on
    // the new runtime.
    // There seems to be no easy way to around this
    // that also avoids deadlocks? (see comment below)
    runtime *old_rt = _rt.exchange(new runtime{});
    if(old_rt)
      delete old_rt;
  }

  runtime *get_runtime() {
    return _rt.load();
  }

  static rt_manager& get() {
    static rt_manager mgr;
    return mgr;
  }


private:
  rt_manager() {
    _rt.store(new runtime{});
  }
  // We cannot use a mutex since this can easily lead to a deadlock:
  // during destruction of the runtime, the destructor waits for
  // the async worker threads (processing scheduling) to finish.
  // The scheduler however also needs to access the runtime to do its work
  // -> deadlock
  std::atomic<runtime*> _rt;
};

class global_settings
{
public:
  static global_settings& get() {
    static global_settings g;
    return g;
  }

  settings &get_settings() {
    return _settings;
  }
private:
  global_settings() {}
  settings _settings;
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

backend_manager &application::backends() {
  return get_runtime().backends();
}

void application::reset() { rt_manager::get().reset(); }

settings &application::get_settings() {
  return global_settings::get().get_settings();
}



}
}

