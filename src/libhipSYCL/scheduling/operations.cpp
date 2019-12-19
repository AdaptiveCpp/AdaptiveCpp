/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#include "CL/sycl/detail/scheduling/operations.hpp"
#include "CL/sycl/detail/scheduling/dag_node.hpp"

namespace cl {
namespace sycl {
namespace detail {


kernel_operation::kernel_operation(kernel_launcher launcher, 
                  const std::vector<memory_requirement*>& memory_requirements)
  : _launcher{launcher},
    _memory_requirements{memory_requirements}
{}

kernel_launcher& 
kernel_operation::get_launcher()
{ return _launcher; }

const kernel_launcher& 
kernel_operation::get_launcher() const
{ return _launcher; }

const std::vector<memory_requirement*>& 
kernel_operation::get_memory_requirements() const
{ return _memory_requirements; }



void requirements_list::add_requirement(std::unique_ptr<requirement> req)
{
  auto node = std::make_shared<dag_node>(
    execution_hints{}, 
    std::vector<dag_node_ptr>{},
    std::move(req));
  
  _reqs.push_back(node);
}

const std::vector<dag_node_ptr>& requirements_list::get() const
{ return _reqs; }

memory_location::memory_location(
    device_id d, sycl::id<3> access_offset,
    std::shared_ptr<buffer_data_region> data_region)
    : _dev{d}, _offset{access_offset},
      _allocation_shape{data_region->get_num_elements()},
      _element_size{data_region->get_element_size()}, _has_data_region{true},
      _data_region{data_region} 
{}


memory_location::memory_location(device_id d, void *base_ptr,
                                 sycl::id<3> access_offset,
                                 sycl::range<3> allocation_shape,
                                 std::size_t element_size)
    : _dev{d}, _offset{access_offset}, _allocation_shape{allocation_shape},
      _element_size{element_size}, _has_data_region{false}, _raw_data{base_ptr}
{}

device_id memory_location::get_device() const
{ return _dev; }

sycl::id<3> memory_location::get_access_offset() const { return _offset; }

sycl::range<3> memory_location::get_allocation_shape() const
{ return _allocation_shape; }

std::size_t memory_location::get_element_size() const { return _element_size; }


bool memory_location::has_data_region() const
{ return _has_data_region; }

bool memory_location::has_raw_pointer() const
{ return !_has_data_region; }

void *memory_location::get_base_ptr() const {
  if (_has_data_region) {
    if (_data_region->has_allocation(_dev)) {
      return _data_region->get_memory(_dev);
    }
    else
      return nullptr;
  } else
    return _raw_data;
}

void *memory_location::get_access_ptr() const {
  void *base_ptr = this->get_base_ptr();

  if (base_ptr == nullptr)
    return nullptr;

  return static_cast<void *>(
      static_cast<char *>(base_ptr) +
      _element_size *
          (_offset[2] + _offset[1] * _allocation_shape[2] +
           _offset[0] * _allocation_shape[1] * _allocation_shape[2]));
  
  
}

memcpy_operation::memcpy_operation(const memory_location &source,
                                   const memory_location &dest,
                                   sycl::range<3> num_source_elements)
    : _source{source}, _dest{dest}, _num_elements{num_source_elements} {}


std::size_t memcpy_operation::get_num_transferred_bytes() const
{
  return _source.get_element_size() * _num_elements.size();
}

const memory_location& memcpy_operation::source() const { return _source; }

const memory_location& memcpy_operation::dest() const { return _dest; }

bool memcpy_operation::is_data_transfer() const { return true; }

}
}
}
