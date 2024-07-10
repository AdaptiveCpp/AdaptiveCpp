/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/instrumentation.hpp"

namespace hipsycl {
namespace rt {

instrumentation_set &operation::get_instrumentations() {
  return _instr_set;
}

const instrumentation_set &operation::get_instrumentations() const {
  return _instr_set;
}

kernel_operation::kernel_operation(
    const char* kernel_name,
    kernel_launcher&& launcher,
    const requirements_list &reqs)
    : _kernel_name{kernel_name}, _launcher{std::move(launcher)} {
  for(auto req_node : reqs.get()){
    operation* op = req_node->get_operation();
    assert(op);
    if(op->is_requirement()){
      requirement* req = cast<requirement>(op);
      if(req->is_memory_requirement()){
        _requirements.push_back(req_node);
      }
    }
  }
}

kernel_launcher& 
kernel_operation::get_launcher()
{ return _launcher; }

const kernel_launcher& 
kernel_operation::get_launcher() const
{ return _launcher; }



void requirements_list::add_requirement(std::unique_ptr<requirement> req)
{
  auto node = std::make_shared<dag_node>(
    execution_hints{}, 
    node_list_t{},
    std::move(req),
    _rt);
  
  add_node_requirement(node);
}

void requirements_list::add_node_requirement(dag_node_ptr node)
{
  // Don't store invalid requirements
  if(node)
    _reqs.push_back(node);
}

const node_list_t& requirements_list::get() const
{ return _reqs; }

memory_location::memory_location(
    device_id d, id<3> access_offset,
    std::shared_ptr<buffer_data_region> data_region)
    : _dev{d}, _offset{access_offset},
      _allocation_shape{data_region->get_num_elements()},
      _element_size{data_region->get_element_size()}, _has_data_region{true},
      _data_region{data_region} 
{}


memory_location::memory_location(device_id d, void *base_ptr,
                                 id<3> access_offset,
                                 range<3> allocation_shape,
                                 std::size_t element_size)
    : _dev{d}, _offset{access_offset}, _allocation_shape{allocation_shape},
      _element_size{element_size}, _has_data_region{false}, _raw_data{base_ptr}
{}

device_id memory_location::get_device() const
{ return _dev; }

id<3> memory_location::get_access_offset() const { return _offset; }

range<3> memory_location::get_allocation_shape() const
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
    } else {
      register_error(
          __acpp_here(),
          error_info{"memory_location: Was configured as data_region-based "
                     "memory location, but data_region did not have allocation "
                     "on the requested device"});
      return nullptr;
    }
  } else
    return _raw_data;
}

void *memory_location::get_access_ptr() const {
  void *base_ptr = this->get_base_ptr();

  if (!base_ptr)
    return nullptr;

  return static_cast<void *>(
      static_cast<char *>(base_ptr) +
      _element_size *
          (_offset[2] + _offset[1] * _allocation_shape[2] +
           _offset[0] * _allocation_shape[1] * _allocation_shape[2]));
  
  
}

memcpy_operation::memcpy_operation(const memory_location &source,
                                   const memory_location &dest,
                                   range<3> num_source_elements)
    : _source{source}, _dest{dest}, _num_elements{num_source_elements} {}


std::size_t memcpy_operation::get_num_transferred_bytes() const
{
  return _source.get_element_size() * _num_elements.size();
}

range<3> memcpy_operation::get_num_transferred_elements() const {
  return this->_num_elements;
}

const memory_location& memcpy_operation::source() const { return _source; }

const memory_location& memcpy_operation::dest() const { return _dest; }

bool memcpy_operation::is_data_transfer() const { return true; }

}
}
