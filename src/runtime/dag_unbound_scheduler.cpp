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
          __acpp_here(),
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

