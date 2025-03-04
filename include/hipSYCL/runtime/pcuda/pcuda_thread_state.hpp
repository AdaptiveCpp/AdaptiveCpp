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

#ifndef ACPP_RT_PCUDA_THREAD_STATE_HPP
#define ACPP_RT_PCUDA_THREAD_STATE_HPP

#include <optional>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "pcuda_stream.hpp"


namespace hipsycl::rt::pcuda {

class pcuda_runtime;

class thread_local_state {
public:
  ~thread_local_state();

  struct kernel_call_configuration {
    dim3 grid;
    dim3 block;
    std::size_t shared_mem;
    pcudaStream_t stream;
  };

  thread_local_state(pcuda_runtime* rt);
  thread_local_state(const thread_local_state&) = delete;
  thread_local_state& operator=(const thread_local_state&) = delete;

  bool set_device(int dev);
  bool set_platform(int platform);
  bool set_backend(int backend);

  int get_device() const;
  int get_platform() const;
  int get_backend() const;

  pcuda::stream* get_default_stream() const;

  void push_kernel_call_config(const kernel_call_configuration& config);
  kernel_call_configuration pop_kernel_call_config();

private:
  pcuda_runtime* _rt;

  int _current_device;
  int _current_platform;
  int _current_backend;

  struct per_device_data {
    std::optional<pcuda::stream*> default_stream;
  };

  mutable std::vector<std::vector<std::vector<per_device_data>>>
      _per_device_data;

  std::optional<kernel_call_configuration> _current_call_config;

};

}

#endif
