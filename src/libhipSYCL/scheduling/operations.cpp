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


memcpy_operation1d::memcpy_operation1d(
    device_id source_device, 
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t element_size,
    sycl::range<1> num_elements)
  : _source_dev{source_device},
    _dest_dev{dest_device},
    _src{src_ptr},
    _dest{dest_ptr},
    _element_size{element_size},
    _count{num_elements}
{}


std::size_t memcpy_operation1d::get_num_dimensions() const
{ return 1; }

std::size_t memcpy_operation1d::get_transferred_size() const
{ return _count[0] * _element_size; }

device_id memcpy_operation1d::get_source_device() const
{ return _source_dev; }

device_id memcpy_operation1d::get_dest_device() const
{ return _dest_dev; }

void* memcpy_operation1d::get_dest_memory() const
{ return _dest; }

const void* memcpy_operation1d::get_source_memory() const
{ return _src; }



memcpy_operation2d::memcpy_operation2d(
    device_id source_device,
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t src_pitch_in_bytes,
    std::size_t dest_pitch_in_bytes,
    std::size_t element_size,
    sycl::range<2> num_elements)
  : _source_dev{source_device},
    _dest_dev{dest_device},
    _src{src_ptr},
    _dest{dest_ptr},
    _src_pitch{src_pitch_in_bytes},
    _dest_pitch{dest_pitch_in_bytes},
    _element_size{element_size},
    _count{num_elements}
{}


std::size_t memcpy_operation2d::get_num_dimensions() const
{ return 2; }

std::size_t memcpy_operation2d::get_transferred_size() const
{ return _count[0] * _count[1] * _element_size; }

device_id memcpy_operation2d::get_source_device() const
{ return _source_dev; }

device_id memcpy_operation2d::get_dest_device() const
{ return _dest_dev; }

void* memcpy_operation2d::get_dest_memory() const
{ return _dest; }

const void* memcpy_operation2d::get_source_memory() const
{ return _src; }


memcpy_operation3d::memcpy_operation3d(
    device_id source_device,
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t element_size,
    sycl::range<3> source_buffer_range,
    sycl::range<3> dest_buffer_range,
    sycl::range<3> num_elements)
  : _source_dev{source_device},
    _dest_dev{dest_device},
    _src{src_ptr},
    _dest{dest_ptr},
    _element_size{element_size},
    _source_buffer_range{source_buffer_range},
    _dest_buffer_range{dest_buffer_range},
    _count{num_elements}
{}

std::size_t memcpy_operation3d::get_num_dimensions() const
{ return 3; }

std::size_t memcpy_operation3d::get_transferred_size() const
{ return _count[0] * _count[1] * _count[2] * _element_size; }

device_id memcpy_operation3d::get_source_device() const
{ return _source_dev; }

device_id memcpy_operation3d::get_dest_device() const
{ return _dest_dev; }

void* memcpy_operation3d::get_dest_memory() const
{ return _dest; }

const void* memcpy_operation3d::get_source_memory() const
{ return _src; }

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



}
}
}
