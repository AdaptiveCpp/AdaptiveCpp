#include "hipSYCL/sycl/serialization/serialization.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/range.hpp"

#include <ostream>
#include <map>

namespace hipsycl::sycl::access {
std::ostream &operator<<(std::ostream &out, const sycl::access::mode value)
{
  switch (value) {
  case sycl::access::mode::read:
    out << "R";
    break;
  case sycl::access::mode::write:
    out << "W";
    break;
  case sycl::access::mode::atomic:
    out << "atomic";
    break;
  case sycl::access::mode::read_write:
    out << "RW";
    break;
  case sycl::access::mode::discard_write:
    out << "Discard W";
    break;
  case sycl::access::mode::discard_read_write:
    out << "Discard RW";
    break;
  default:
    throw "Mode enum cannot be serialized";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const sycl::access::target value)
{
  switch (value) {
  case sycl::access::target::image:
    out << "image";
    break;
  case sycl::access::target::constant_buffer:
    out << "constant_buffer";
    break;
  case sycl::access::target::global_buffer:
    out << "global_buffer";
    break;
  case sycl::access::target::host_buffer:
    out << "host_buffer";
    break;
  case sycl::access::target::host_image:
    out << "host_image";
    break;
  case sycl::access::target::image_array:
    out << "Image_array";
    break;
  case sycl::access::target::local:
    out << "local";
    break;
  default:
    throw "Target enum cannot be serialized";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out,
                         const sycl::access::placeholder value)
{
  switch (value) {
  case sycl::access::placeholder::false_t:
    out << "false";
    break;
  case sycl::access::placeholder::true_t:
    out << "true";
    break;
  default:
    throw "Placeholder enum cannot be serialized";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out,
                         const sycl::access::fence_space value) 
{
  switch (value) {
  case sycl::access::fence_space::global_and_local:
    out << "global and local";
    break;
  case sycl::access::fence_space::global_space:
    out << "global";
    break;
  case sycl::access::fence_space::local_space:
    out << "local";
    break;
  default:
    throw "fence_space enum cannot be serialized";
    break;
  }
  return out;
}

} // namespace hipsycl::sycl::access
