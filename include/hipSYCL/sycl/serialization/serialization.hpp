#ifndef HIPSYCL_DUMP_INTERFACE_HPP
#define HIPSYCL_DUMP_INTERFACE_HPP

#include "hipSYCL/sycl.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/range.hpp"

#include <ostream>
#include <map>

// The << operator is implemented for the sycl interface classes
// and enums. For runtime classes and structs a dump member function
// is implemented
namespace hipsycl::sycl::access {
std::ostream &operator<<(std::ostream &out, const sycl::access::mode value);
std::ostream &operator<<(std::ostream &out, const sycl::access::target value);
std::ostream &operator<<(std::ostream &out,
                         const sycl::access::placeholder value);
std::ostream &operator<<(std::ostream &out,
                         const sycl::access::fence_space value);
} // namespace hipsycl::sycl::access

namespace hipsycl::sycl {

template <int dimensions>
std::ostream &operator<<(std::ostream &out, const id<dimensions> id)
{
  out << "(";
  for (int i = 0; i < dimensions - 1; i++) {
    out << id[i] << ',';
  }
  out << id[dimensions - 1] << ")";
  return out;
}

template <int dimensions>
std::ostream &operator<<(std::ostream &out, const range<dimensions> range)
{
  out << "(";
  for (int i = 0; i < dimensions - 1; i++) {
    out << range[i] << ',';
  }
  out << range[dimensions - 1] << ")";
  return out;
}

} // namespace hipsycl::sycl
#endif
