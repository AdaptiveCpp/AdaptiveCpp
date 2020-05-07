#ifndef HIPSYCL_DUMP_RUNTIME_HPP
#define HIPSYCL_DUMP_RUNTIME_HPP

#define HIPSYCL_DUMP_INDENTATION "   "
#include "hipSYCL/runtime/backend.hpp"

#include <ostream>
#include <map>

namespace hipsycl::rt {
std::ostream &operator<<(std::ostream &out, const hardware_platform value);
std::ostream &operator<<(std::ostream &out, const api_platform value);
std::ostream &operator<<(std::ostream &out, const backend_id value);
} // namespace hipsycl::rt

#endif
