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

#include "hipSYCL/runtime/pcuda/pcuda_device_topology.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include <cstddef>

namespace hipsycl::rt::pcuda {

device_topology::device_topology(runtime *rt) {
  int backend_idx = 0;
  rt->backends().for_each_backend([&, this](rt::backend *b) {
    device_topology::backend backend_descriptor;
    backend_descriptor.backend_ptr = b;
    backend_descriptor.pcuda_backend_index = backend_idx;
    backend_descriptor.id = b->get_unique_backend_id();

    for (int platform = 0;
         platform < b->get_hardware_manager()->get_num_platforms();
         ++platform) {
      device_topology::platform plat;
      plat.pcuda_platform_id = platform;

      int pcuda_dev_idx = 0;
      for (int global_dev_idx = 0;
           global_dev_idx < b->get_hardware_manager()->get_num_devices();
           ++global_dev_idx) {
        auto *d = b->get_hardware_manager()->get_device(global_dev_idx);
        if (d->get_platform_index() == platform) {
          ++pcuda_dev_idx;

          device_topology::device dev;
          dev.pcuda_device_id = pcuda_dev_idx;
          dev.rt_device_id =
              rt::device_id{b->get_backend_descriptor(), global_dev_idx};
          dev.dev = d;

          plat.devices.push_back(dev);
        }
      }

      backend_descriptor.platforms.push_back(plat);
    }
    backends.push_back(backend_descriptor);

    ++backend_idx;
  });

  for(int i = 0; i < backends.size(); ++i) {
    HIPSYCL_DEBUG_INFO << "[PCUDA] device_topology: Backend " << i << " ("
                       << backends[i].backend_ptr->get_name() << ")\n";
    for(int p = 0; p < backends[i].platforms.size(); ++p) {
      HIPSYCL_DEBUG_INFO << "[PCUDA] device_topology:   Platform " << p << "\n";
      for(int d = 0; d < backends[i].platforms[p].devices.size(); ++d) {
        HIPSYCL_DEBUG_INFO
            << "[PCUDA] device_topology:   Device " << d << ": "
            << backends[i].platforms[p].devices[d].dev->get_device_name()
            << "\n";
      }
    }
  }
}

const device_topology::backend *
device_topology::get_backend(int backend_idx) const {
  if (backend_idx < 0 || backend_idx >= backends.size())
    return nullptr;
  return &(backends[backend_idx]);
}

const device_topology::platform *
device_topology::get_platform(int backend_idx, int platform_idx) const {
  auto* b = get_backend(backend_idx);
  if(!b)
    return nullptr;

  if(platform_idx < 0 || platform_idx >= b->platforms.size())
    return nullptr;

  return &(b->platforms[platform_idx]);
}

const device_topology::device *
device_topology::get_device(int backend_index, int platform_idx,
                            int device_idx) const {
  auto *p = get_platform(backend_index, platform_idx);
  if (!p)
    return nullptr;

  if (device_idx < 0 || device_idx >= p->devices.size())
    return nullptr;

  return &(p->devices[device_idx]);
}

bool device_topology::device_id_to_index_triple(device_id dev,
                                                int &backend_idx_out,
                                                int &platform_idx_out,
                                                int &device_idx_out) const {
  for(backend_idx_out = 0; backend_idx_out < backends.size(); ++backend_idx_out) {
    if(backends[backend_idx_out].id == dev.get_backend()) {
      auto& platforms = backends[backend_idx_out].platforms;
      for(platform_idx_out = 0; platform_idx_out < platforms.size(); ++platform_idx_out) {
        auto& devs = platforms[platform_idx_out].devices;
        for(device_idx_out = 0; device_idx_out < devs.size(); ++device_idx_out) {
          if(devs[device_idx_out].rt_device_id == dev) {
            return true;
          }
        }
      }
    }
  }

  return false;
}
}
