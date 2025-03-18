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

#ifndef ACPP_RT_PCUDA_DEVICE_TOPOLOGY_HPP
#define ACPP_RT_PCUDA_DEVICE_TOPOLOGY_HPP

#include "hipSYCL/runtime/device_id.hpp"

namespace hipsycl::rt {

class runtime;
class hardware_context;
class backend;

}

namespace hipsycl::rt::pcuda {

class device_topology {
public:
  device_topology(runtime* rt);


  struct device {
    int pcuda_device_id;
    device_id rt_device_id;
    hardware_context* dev;
  };

  struct platform {
    int pcuda_platform_id;
    std::vector<device_topology::device> devices;
  };

  struct backend {
    int pcuda_backend_index;
    backend_id id;
    rt::backend* backend_ptr;
    std::vector<device_topology::platform> platforms;
  };

  const device_topology::backend *get_backend(int backend_idx) const;
  const device_topology::platform *get_platform(int backend_idx,
                                                int platform_idx) const;
  const device_topology::device *get_device(int backend_index, int platform_idx,
                                            int device_idx) const;

  bool device_id_to_index_triple(device_id dev, int &backend_idx_out,
                                 int &platform_idx_out,
                                 int &device_idx_out) const;

  const std::vector<device_topology::backend>& all_backends() const {
    return backends;
  }
private:
  std::vector<device_topology::backend> backends;
};

}

#endif
