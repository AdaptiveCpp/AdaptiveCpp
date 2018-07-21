#ifndef SYCU_CONTEXT_HPP
#define SYCU_CONTEXT_HPP

#include "types.hpp"
#include "exception.hpp"
#include "device.hpp"
#include "platform.hpp"

#include <cassert>

namespace cl {
namespace sycl {

class context
{
public:
  explicit context(async_handler asyncHandler = {});

  context(const device &dev, async_handler asyncHandler = {})
    : _platform{dev.get_platform()}, _devices{dev}
  {}

  context(const platform &plt, async_handler asyncHandler = {})
    : _platform{plt}, _devices(plt.get_devices())
  {}

  context(const vector_class<device> &deviceList,
          async_handler asyncHandler = {})
    : _devices{deviceList}
  {
    if(deviceList.empty())
      throw platform_error{"context: Could not infer platform from empty device list"};

    _platform = deviceList.front();

    for(const auto dev : deviceList)
      assert(dev.get_platform() == _platform);
  }

  /* CL Interop is not supported
  context(cl_context clContext, async_handler asyncHandler = {});
  */


  /* -- common interface members -- */


  /* CL interop is not supported
  cl_context get() const;
*/

  bool is_host() const {
    return false;
  }

  platform get_platform() const {
    return _platform;
  }

  vector_class<device> get_devices() const {
    return _devices;
  }

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type get_info() const {
    static_assert(false, "Unimplemented");
  }

private:
  platform _platform;
  vector_class<device> _devices;
};

} // namespace sycl
} // namespace cl



#endif
