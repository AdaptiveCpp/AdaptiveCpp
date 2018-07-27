#ifndef SYCU_DEVICE_SELECTOR_HPP
#define SYCU_DEVICE_SELECTOR_HPP

#include "exception.hpp"
#include "device.hpp"

#include <limits>

namespace cl {
namespace sycl {


class device_selector
{
public:
  virtual ~device_selector();
  device select_device() const;

  virtual int operator()(const device& dev) const = 0;

};


class gpu_selector : public device_selector
{
public:
  virtual ~gpu_selector(){}
  virtual int operator()(const device& dev) const
  {
    return 1;
  }
};

class error_selector : public device_selector
{
public:
  virtual ~error_selector(){}
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
