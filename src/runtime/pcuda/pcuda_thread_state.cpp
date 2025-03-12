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

#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"

#include <unordered_map>
#include <cstdint>

namespace hipsycl::rt::pcuda {

thread_local_state::~thread_local_state() {
  for(auto& backend : _per_device_data){
    for(auto& platform : backend) {
      for(auto& device : platform) {
        if(device.default_stream.has_value()) {
          auto err = stream::destroy(device.default_stream.value(), _rt);
          if(err != pcudaSuccess) {
            register_pcuda_error(__acpp_here(), err, "default stream destruction failed");
          }
        }
      }
    }
  }
}

thread_local_state::thread_local_state(pcuda_runtime* rt)
: _rt{rt}, _current_backend{0}, _current_platform{0}, _current_device{0} {

  auto& topo = rt->get_topology();
  
  int best_backend = 0;
  int best_platform = 0;
  int best_score = -1;

  for(int i = 0; i < topo.all_backends().size(); ++i) {
    backend* b = topo.get_backend(i)->backend_ptr;

    if(b->get_hardware_manager()->get_num_devices() > 0) {
      for(int j = 0; j < topo.get_backend(i)->platforms.size(); ++j) {
        int platform_score = 0;

        for(int k = 0; k < topo.get_platform(i, j)->devices.size(); ++k) {
          auto* dev = topo.get_device(i, j, k)->dev;
          auto dev_id = topo.get_device(i, j, k)->rt_device_id;

          if(dev->is_cpu()) {
            // Prefer OpenCL CPU device over OpenMP one
            // (users can always set ACPP_VISIBILITY_MASK to force selection of
            // OpenMP, while the reverse would not be possible without this 
            // preference)
            if(dev_id.get_backend() == backend_id::omp)
              platform_score += 1;
            else
              platform_score += 2;
          } else if(dev->is_gpu()){
            // Always prefer GPU.
            // Note that we *add* scores, so a platform with more devices is
            // always preferred
            
            // Prefer CUDA, since a) CUDA tends to be the most reliable backend
            // and b) we know that the hardware is going to be a dGPU
            if(dev_id.get_backend() == backend_id::cuda)
              platform_score += 6;
            // HIP is typically a dGPU, but might also be an APU
            else if(dev_id.get_backend() == backend_id::hip)
              platform_score += 5;
            else {
              // OpenCL or L0 is most likely iGPU. Not many Intel dGPUs around.
              platform_score += 4;
            }
          } else {
            // not a CPU nor GPU? Such a device is currently not tested with
            // AdaptiveCpp, be cautious and prefer any other platform.
            platform_score += 0;
          }
        }

        if(platform_score > best_score) {
          best_backend = i;
          best_platform = j;
          best_score = platform_score;
        }
      }
    }
  }

  if(best_score < 0) {
    // Cannot register as this happens typically during startup
    print_warning(__acpp_here(),
                  error_info{"[PCUDA] pcuda_thread_state: Did not find any "
                             "devices (not even CPU); this should "
                             "never happen. Things are going to break now."});
  } else {
    _current_backend = best_backend;
    _current_platform = best_platform;
    _current_device = 0;
  }
  HIPSYCL_DEBUG_INFO << "[PCUDA] thread_state: Default device: backend "
                     << _current_backend << ", platform " << _current_platform
                     << ", device " << _current_device << std::endl;

  _per_device_data.resize(topo.all_backends().size());
  for(int i = 0; i < _per_device_data.size(); ++i) {
    _per_device_data[i].resize(topo.get_backend(i)->platforms.size());
    for(int j = 0; j < _per_device_data[i].size(); ++j) {
      _per_device_data[i][j].resize(topo.get_platform(i, j)->devices.size());
    }
  }

}

int thread_local_state::get_device() const { return _current_device; }
int thread_local_state::get_platform() const { return _current_platform; }
int thread_local_state::get_backend() const { return _current_backend; }

pcuda::stream* thread_local_state::get_default_stream() const {
  if(_current_backend >= _per_device_data.size())
    return nullptr;
  if(_current_platform >= _per_device_data[_current_backend].size())
    return nullptr;
  if(_current_device >=
         _per_device_data[_current_backend][_current_platform].size())
    return nullptr;

  auto &device_data =
      _per_device_data[_current_backend][_current_platform][_current_device];


  if(pcuda::stream* s = device_data.default_stream.value_or(nullptr))
    return s;
  
  pcuda::stream* default_stream = nullptr;
  auto *dev = _rt->get_topology().get_device(
      _current_backend, _current_platform, _current_device);
  assert(dev);
  auto err = stream::create(default_stream, _rt, dev->rt_device_id, 0, 0);

  if(err != pcudaSuccess) {
    register_pcuda_error(__acpp_here(), err,
                         "default stream construction failed");
    return nullptr;
  }
  assert(default_stream);
  device_data.default_stream = default_stream;

  return default_stream;
}

void thread_local_state::push_kernel_call_config(const kernel_call_configuration& config) {
  if(_current_call_config.has_value()) {
    HIPSYCL_DEBUG_WARNING
        << "[PCUDA] thread_local_state: Pushing new call configuration, but "
           "the previous call configuration has not yet been popped. This "
           "indicates a prior incomplete kernel launch and should not happen."
        << std::endl;
  }
  _current_call_config = config;
}

thread_local_state::kernel_call_configuration
thread_local_state::pop_kernel_call_config() {
  if(!_current_call_config.has_value()) {
    register_pcuda_error(__acpp_here(), pcudaErrorMissingConfiguration,
                         "thread_local_state: Could not pop kernel "
                         "launch configuration. The kernel launch was likely "
                         "not configured prior to launch.");
    return kernel_call_configuration{};
  }
  auto value = _current_call_config.value();
  _current_call_config.reset();
  return value;
}

bool thread_local_state::set_device(int dev) {
  auto &devs = _rt->get_topology()
                   .get_platform(_current_backend, _current_platform)
                   ->devices;
  if(dev < devs.size()) {
    _current_device = dev;
    return true;
  }
  return false;
}

bool thread_local_state::set_platform(int platform) {
  auto &platforms = _rt->get_topology()
                   .get_backend(_current_backend)
                   ->platforms;
  if(platform < platforms.size()) {
    _current_platform = platform;
    return true;
  }
  return false;
}

bool thread_local_state::set_backend(int backend) {
  auto& backends = _rt->get_topology().all_backends();
  if(backend < backends.size()) {
    _current_backend = backend;
    return true;
  }
  return false;
}

}
