#include "CL/sycl/device.hpp"
#include "CL/sycl/device_selector.hpp"
#include "CL/sycl/platform.hpp"

namespace cl {
namespace sycl {


device::device(const device_selector &deviceSelector) {
  this->_device_id = deviceSelector.select_device()._device_id;
}

platform device::get_platform() const  {
  // We only have one platform
  return platform{};
}

vector_class<device> device::get_devices(
    info::device_type deviceType)
{
  if(deviceType == info::device_type::cpu ||
     deviceType == info::device_type::host)
    return vector_class<device>();

  vector_class<device> result;
  int num_devices = get_num_devices();
  for(int i = 0; i < num_devices; ++i)
  {
    device d;
    d._device_id = i;

    result.push_back(d);
  }
  return result;
}

int device::get_num_devices()
{
  int num_devices = 0;
  detail::check_error(hipGetDeviceCount(&num_devices));
  return num_devices;
}

int device::get_device_id() const {
  return _device_id;
}

namespace detail {

void set_device(const device& d) {
  detail::check_error(hipSetDevice(d.get_device_id()));
}

}

}
}
