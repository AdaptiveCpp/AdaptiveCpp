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

  explicit device(const device_selector &deviceSelector) {
    this->_device_id = deviceSelector.select_device()._device_id;
  }

  // OpenCL interop is not supported
  // cl_device_id get() const;

  bool is_host() const {return false;}

  bool is_cpu() const {return false; }

  bool is_gpu() const {return true; }

  bool is_accelerator() const {return true; }

  platform get_platform() const {
    // We only have one platform
    return platform{};
  }

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const
  {
    static_assert(false, "Unimplemented");
  }

  bool has_extension(const string_class &extension) const;

  // Available only when prop == info::partition_property::partition_equally
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(size_t nbSubDev) const;

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(const vector_class<size_t> &counts) const
  {
    static_assert(false, "Unimplemented");
  }

  // Available only when prop == info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(info::affinity_domain affinityDomain) const
  {
    static_assert(false, "Unimplemented");
  }

  static vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all)
  {
    if(type == info::device_type::cpu ||
       type == info::device_type::host ||
       type == info::device_type::opencl)
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

  static int get_num_devices() {
    int num_devices = 0;
    detail::check_error(hipGetDeviceCount(&num_devices));
    return num_devices;
  }

  int get_device_id() const {
    return _device_id;
  }
private:
  int _device_id;
};


template <>
inline auto device::get_info<info::device::device_type>() const {
  return info::device_type::gpu;
}

template <>
inline auto device::get_info<info::device::vendor_id>() const {
  // ToDo: Calculate unique vendor id
  return 0;
}

namespace detail {

static void set_device(const device& d) {
  detail::check_error(hipSetDevice(d.get_device_id()));
}

}


} // namespace sycl
} // namespace cl



#endif
