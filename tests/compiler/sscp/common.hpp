
#pragma once

#include <vector>
#include <sycl/sycl.hpp>

inline sycl::queue get_queue() {
  std::vector<sycl::device> sscp_devices;

  for(const auto& dev : sycl::device::get_devices()) {
    auto dev_id = dev.AdaptiveCpp_device_id();
    auto* rt = dev.AdaptiveCpp_runtime();

    if(rt->backends().get(dev_id.get_backend())
                   ->get_hardware_manager()
                   ->get_device(dev_id.get_id())->has(hipsycl::rt::device_support_aspect::sscp_kernels)) {
      sscp_devices.push_back(dev);
    }
  }

  if(sscp_devices.size() == 0)
    throw std::runtime_error{"No suitable device was found"};

  for(const auto& dev : sscp_devices)
    if(!dev.is_cpu())
      return sycl::queue{dev};
  for(const auto& dev : sscp_devices)
    if (dev.AdaptiveCpp_device_id().get_backend() != hipsycl::rt::backend_id::omp)
      return sycl::queue{dev};

  return sycl::queue{sscp_devices[0]};
}
