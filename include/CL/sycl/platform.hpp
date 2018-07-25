#ifndef SYCU_PLATFORM_HPP
#define SYCU_PLATFORM_HPP

#include "types.hpp"
#include "device_selector.hpp"
#include "info/platform.hpp"

namespace cl {
namespace sycl {

class device_selector;

class platform {

public:

  platform() {}

  /* OpenCL interop is not supported
  explicit platform(cl_platform_id platformID);
  */

  explicit platform(const device_selector &deviceSelector) {}


  /* -- common interface members -- */

  /* OpenCL interop is not supported
  cl_platform_id get() const;
  */


  vector_class<device> get_devices(
      info::device_type type = info::device_type::all) const
  {
    return device::get_devices();
  }


  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type get_info() const
  {
    static_assert(false, "Unimplemented");
  }


  /// \todo Think of a better solution
  bool has_extension(const string_class &extension) const {
    return false;
  }


  bool is_host() const {
    return false;
  }


  static vector_class<platform> get_platforms() {
    return vector_class<platform>{platform()};
  }

};

}// namespace sycl
}// namespace cl

#endif
