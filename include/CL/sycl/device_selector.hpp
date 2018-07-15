#ifndef SYCU_DEVICE_SELECTOR_HPP
#define SYCU_DEVICE_SELECTOR_HPP

#include "device.hpp"
#include "exception.hpp"
#include <limits>

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
  if(devices.size() == 0)
    throw platform_error{"No available devices!"};

  int best_score = std::numeric_limits<int>::min();
  device candidate;
  for(const device& d : devices)
  {
    int current_score = (*this)(d);
    if(d > best_score)
    {
      best_score = current_score;
      candidate = d;
    }
  }
  return candidate;
}

class gpu_selector : public device_selector
{
public:
  virtual ~default_selector(){}
  virtual int operator()(const device& dev) const
  {
    return 1;
  }
};

class error_selector : public device_selector
{
public:
  virtual ~default_selector(){}
  virtual int operator()(const device& dev) const
  {
    throw unimplemented{"SYCU presently only supports GPU platforms and device selectors."};
  }
};

using default_selector = gpu_selector;
using cpu_selector = error_selector;
using host_selector = error_selector;

}  // namespace sycl
}  // namespace cl

#endif
