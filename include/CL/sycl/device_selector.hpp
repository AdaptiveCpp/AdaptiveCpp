#ifndef SYCU_DEVICE_SELECTOR_HPP
#define SYCU_DEVICE_SELECTOR_HPP

#include "device.hpp"
#include "exception.hpp"

namespace cl {
namespace sycl {

class device_selector
{
public:
  virtual ∼device_selector();
  device select_device() const;

  virtual int operator()(const device& dev) const = 0;

};


device device_selector::select_device()
{
  auto devices = device::get_devices();

  for(const device& d : devices)

}



}  // namespace sycl
}  // namespace cl

#endif
