/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/dag.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/util.hpp"

#include <map>
#include <ostream>

namespace hipsycl::rt {

std::ostream &operator<<(std::ostream &out, const hardware_platform value) {
  switch (value) {
  case rt::hardware_platform::cpu:
    out << "CPU";
    break;
  case rt::hardware_platform::cuda:
    out << "CUDA";
    break;
  case rt::hardware_platform::rocm:
    out << "ROCm";
    break;
  case rt::hardware_platform::level_zero:
    out << "Level Zero";
    break;
  case rt::hardware_platform::ocl:
    out << "OpenCL";
    break;
  default:
    out << "<unknown>";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const api_platform value) {
  switch (value) {
  case rt::api_platform::hip:
    out << "HIP";
    break;
  case rt::api_platform::omp:
    out << "OpenMP";
    break;
  case rt::api_platform::cuda:
    out << "CUDA";
    break;
  case rt::api_platform::level_zero:
    out << "Level Zero";
    break;
  case rt::api_platform::ocl:
    out << "OpenCL";
    break;
  default:
    out << "<unknown>";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const backend_id value) {
  switch (value) {
  case rt::backend_id::hip:
    out << "HIP";
    break;
  case rt::backend_id::cuda:
    out << "CUDA";
    break;
  case rt::backend_id::omp:
    out << "OpenMP";
    break;
  case rt::backend_id::level_zero:
    out << "Level Zero";
    break;
  case rt::backend_id::ocl:
    out << "OpenCL";
    break;
  default:
    out << "<unknown>";
    break;
  }
  return out;
}

// Implementing dump member function of various classes:
std::string get_indentation(int indentation) {
  std::string indent;
  for (int i = 0; i < indentation; i++) {
    indent += HIPSYCL_DUMP_INDENTATION;
  }
  return indent;
}

// operations
void buffer_memory_requirement::dump(std::ostream &ostr,
                                     int indentation) const {
  ostr << get_indentation(indentation);
  ostr << "MEM_REQ: " << _mode << " " << _target << " " << _offset << "+"
       << _range << " #" << _element_size;
}

void kernel_operation::dump(std::ostream &ostr, int indentation) const {
  std::string indent = get_indentation(indentation);
  ostr << indent << "kernel: " << _kernel_name;
  for (auto requirement : _requirements) {
    ostr << std::endl;
    requirement->get_operation()->dump(ostr, indentation + 1);
  }
}

void memcpy_operation::dump(std::ostream &ostr, int indentation) const {

  ostr << get_indentation(indentation);
  ostr << "Memcpy: ";
  _source.dump(ostr); // Memory location
  ostr << "-->";
  _dest.dump(ostr); // Memory location
  ostr << _num_elements;
}

void prefetch_operation::dump(std::ostream& ostr, int indentation) const {
  ostr << get_indentation(indentation);
  ostr << "Prefetch: " << _num_bytes << " bytes from " << _ptr;
}

void memset_operation::dump(std::ostream &ostr, int indentation) const {
  ostr << get_indentation(indentation);
  ostr << "Memset: @" << _ptr << " " << _num_bytes << " bytes of value "
       << static_cast<int>(get_pattern());
}

void memory_location::dump(std::ostream &ostr) const {
  _dev.dump(ostr);
  ostr << " #" << _element_size << " " << _offset << "+" << _allocation_shape;
}

// dag
void dag::dump(std::ostream &ostr) const {
  for_each_node([&](dag_node_ptr node_ptr) {
    ostr << "Node " << node_ptr << std::endl;
    node_ptr->get_operation()->dump(ostr, 1);
    ostr << HIPSYCL_DUMP_INDENTATION << "Has requirement on: ";
    auto requirement_list = node_ptr->get_requirements();
    if (!requirement_list.empty()) {
      for (auto weak_req : requirement_list) {
        if(auto req = weak_req.lock())
          ostr << req << " ";
      }
    } else {
      std::cout << "None";
    }
  });
}

// device_id

void device_id::dump(std::ostream &ostr) const {
  ostr << _backend.hw_platform << "-Device" << _device_id;
}

std::ostream &operator<<(std::ostream &out, device_id dev) {
  dev.dump(out);
  return out;
}

} // end of namespace hipsycl::rt
