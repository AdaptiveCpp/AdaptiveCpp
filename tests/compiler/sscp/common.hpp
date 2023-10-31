
#pragma once

#include <sycl/sycl.hpp>

inline sycl::queue get_queue() {
  for(const auto& dev : sycl::device::get_devices()) {
    auto dev_id = dev.hipSYCL_device_id();
    auto* rt = dev.hipSYCL_runtime();

    if(rt->backends().get(dev_id.get_backend())
                   ->get_hardware_manager()
                   ->get_device(dev_id.get_id())->has(hipsycl::rt::device_support_aspect::sscp_kernels)) {
      return sycl::queue{dev};  
    }
  }
  throw std::runtime_error{"No suitable device was found"};
}
