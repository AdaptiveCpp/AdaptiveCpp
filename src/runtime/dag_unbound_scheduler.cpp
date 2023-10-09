/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#include "hipSYCL/runtime/dag_unbound_scheduler.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl {
namespace rt {

dag_unbound_scheduler::dag_unbound_scheduler(runtime* rt)
: _direct_scheduler{rt}, _rt{rt} {}

void dag_unbound_scheduler::submit(dag_node_ptr node) {
  if(_devices.empty()) {
    // We cannot query this in the constructor, because
    // when schedulers are constructed the runtime is typically
    // locked because it is just starting up, so this would
    // create a deadlock
    _rt->backends().for_each_backend([this](backend *b) {
      std::size_t num_devs = b->get_hardware_manager()->get_num_devices();
      for (std::size_t i = 0; i < num_devs; ++i) {
        this->_devices.push_back(b->get_hardware_manager()->get_device_id(i));
      }
    });
  }

  if(!node->get_execution_hints().has_hint<hints::bind_to_device>()){
    std::vector<rt::device_id> eligible_devices;
    if(node->get_execution_hints().has_hint<hints::bind_to_device_group>()) {
      eligible_devices = node->get_execution_hints()
                             .get_hint<hints::bind_to_device_group>()
                             ->get_devices();
    } else {
      eligible_devices = _devices;
    }

    if(eligible_devices.empty()) {
      register_error(
          __hipsycl_here(),
          error_info{"dag_unbound_scheduler: No devices available to "
                     "dispatch operation; this indicates that the "
                     "device selector did not find appropriate devices."});
      node->cancel();
      return;
    }
    // Round-robin as placeholder. This is not intended
    // for anything practical, it's a _placeholder_
    static std::size_t dev = 0;
    ++dev;

    rt::device_id target_dev = eligible_devices[dev % eligible_devices.size()];
    node->get_execution_hints().set_hint(rt::hints::bind_to_device{target_dev});
  }

  _direct_scheduler.submit(node);
}

}
}

