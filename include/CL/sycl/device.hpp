#ifndef SYCU_DEVICE_HPP
#define SYCU_DEVICE_HPP

#include "types.hpp"
#include "info/device.hpp"
#include "info/param_traits.hpp"
#include "backend/backend.hpp"
#include "exception.hpp"
#include "id.hpp"

namespace cl {
namespace sycl {
namespace info {

#define SYCU_DEVICE_PARAM_TRAIT(param, return_value) \
  SYCU_PARAM_TRAIT_RETURN_VALUE(info::device, info::device::param, return_value)

SYCU_DEVICE_PARAM_TRAIT(device_type, info::device_type);
SYCU_DEVICE_PARAM_TRAIT(vendor_id, cl_uint);
//SYCU_DEVICE_PARAM_TRAIT(max_compute_units, cl_uint);
//SYCU_DEVICE_PARAM_TRAIT(max_work_item_dimensions, cl_uint);
//SYCU_DEVICE_PARAM_TRAIT(max_work_item_sizes, id<3>);
//SYCU_DEVICE_PARAM_TRAIT(max_work_group_size, std::size_t);
// ToDo: Complete this
} // info

class device_selector;
class platform;

class device
{
public:

  /// Since we do not support host execution, this will actually
  /// try to use the first GPU. Note: SYCL spec requires that
  /// this should actually create a device object for host execution.
  ///
  /// \todo Should this call throw an error instead of behaving differently
  /// than the spec requires?
  device()
    : _device_id{0}
  {}

  // OpenCL interop is not supported
  // explicit device(cl_device_id deviceId);

  explicit device(const device_selector &deviceSelector);

  // OpenCL interop is not supported
  // cl_device_id get() const;

  bool is_host() const {return false;}

  bool is_cpu() const {return false; }

  bool is_gpu() const {return true; }

  bool is_accelerator() const {return true; }

  platform get_platform() const;

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const
  {
    throw unimplemented{"device::get_info() is unimplemented"};
  }

  bool has_extension(const string_class &extension) const
  {
    throw unimplemented{"device::has_extension is unimplemented"};
  }

  /* create_sub_devices is not yet supported

  // Available only when prop == info::partition_property::partition_equally
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(size_t nbSubDev) const;

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(const vector_class<size_t> &counts) const
  {
    throw unimplemented{"device::create_sub_devices is unimplemented"};
  }

  // Available only when prop == info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(info::affinity_domain affinityDomain) const
  {
    throw unimplemented{"device::create_sub_devices is unimplemented"};
  }
  */

  static vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all);

  static int get_num_devices();

  int get_device_id() const;

  bool operator ==(const device& rhs) const
  { return rhs._device_id == _device_id; }

  bool operator !=(const device& rhs) const
  { return !(*this == rhs); }
private:
  int _device_id;
};

/*
template <>
inline auto device::get_info<info::device::device_type>() const {
  return info::device_type::gpu;
}

template <>
inline auto device::get_info<info::device::vendor_id>() const {
  // ToDo: Calculate unique vendor id
  return 0;
}*/

namespace detail {

void set_device(const device& d);

}


} // namespace sycl
} // namespace cl



#endif
