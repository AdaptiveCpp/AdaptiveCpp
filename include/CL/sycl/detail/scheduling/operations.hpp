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

#ifndef HIPSYCL_OPERATIONS_HPP
#define HIPSYCL_OPERATIONS_HPP

#include "../../backend/backend.hpp"
#include "../../id.hpp"
#include "../../range.hpp"
#include "../../access.hpp"

#include "device_id.hpp"
#include "data.hpp"
#include "executor.hpp"

#include <functional>
#include <memory>

namespace cl {
namespace sycl {
namespace detail {

using cost_type = double;

class operation
{
public:
  virtual ~operation(){}

  virtual cost_type get_runtime_costs() { return 1.; }
};


class requirement : public operation
{
public:
  virtual bool is_memory_requirement() const = 0;

  virtual ~requirement(){}
};

class memory_requirement : public requirement
{
public:
  virtual ~memory_requirement() {}

  virtual bool is_memory_requirement() const final override
  { return true; }

  virtual std::size_t get_required_size() const = 0;
  virtual bool is_image_requirement() const = 0;
  virtual int get_dimensions() const = 0;
  virtual std::size_t get_element_size() const = 0;

  virtual sycl::access::mode get_access_mode() const = 0;
  virtual sycl::access::target get_access_target() const = 0;

  virtual sycl::id<3> get_access_offset3d() const = 0;
  virtual sycl::range<3> get_access_range3d() const = 0;

  bool is_buffer_requirement() const 
  { return !is_image_requirement(); }
};


class buffer_memory_requirement : public memory_requirement
{
public:

  template<int Dim>
  buffer_memory_requirement(
    std::shared_ptr<buffer_data_region> mem_region,
    const sycl::id<Dim> offset, 
    const sycl::range<Dim> range,
    std::size_t element_size,
    sycl::access::mode access_mode,
    sycl::access::target access_target)
  : _mem_region{mem_region},
    _element_size{element_size},
    _mode{access_mode},
    _target{access_target},
    _dimensions{Dim}
  {
    static_assert(Dim >= 1 && Dim <=3, 
      "dimension of buffer memory requirement must be between 1 and 3");

    if(Dim == 1){
      _offset = sycl::id<3>{offset[0], 0, 0};
      _range = sycl::range<3>{range[0], 1, 1};
    }
    else if(Dim == 2){
      _offset = sycl::id<3>{offset[0], offset[1], 0};
      _range = sycl::range<3>{range[0], range[1], 1};
    }
    else {
      _offset = offset;
      _range = range;
    }

  }

  std::size_t get_required_size() const override
  {
    std::size_t num_elements = _range[0];
    if(_dimensions > 1)
      num_elements *= _range[1];
    if(_dimensions > 2)
      num_elements *= _range[2];
    return num_elements * _element_size;
  }

  bool is_image_requirement() const override
  { return false; }

  int get_dimensions() const override
  { return _dimensions; }

  std::size_t get_element_size() const override
  { return _element_size; }

  sycl::access::mode get_access_mode() const override
  { return _mode; }

  sycl::access::target get_access_target() const override
  { return _target; }


  std::shared_ptr<buffer_data_region> get_data_region() const
  { return _mem_region; }

  sycl::id<3> get_access_offset3d() const override
  {
    return _offset;
  }

  sycl::range<3> get_access_range3d() const override
  {
    return _range;
  }
private:
  std::shared_ptr<buffer_data_region> _mem_region;
  sycl::id<3> _offset;
  sycl::range<3> _range;
  std::size_t _element_size;
  sycl::access::mode _mode;
  sycl::access::target _target;
  int _dimensions;
};

class kernel_launcher
{
public:
  // Parameters are: device_id, stream
  using stream_launch_functor = std::function<void(device_id, backend_executor*)>;

  // TODO: In the future, we could have a templated constructor and submit lambdas
  // and let the constructor here construct the actual invocation functors
  kernel_launcher(stream_launch_functor hip_launch, stream_launch_functor host_launch);

  void invoke_hip_kernel(device_id dev, backend_executor* q);
  void invoke_host_kernel(device_id dev, backend_executor* q);

private:
  stream_launch_functor _hip_launcher;
  stream_launch_functor _host_launcher;
};

class kernel_operation : public operation
{
public:
  kernel_operation(kernel_launcher launcher, 
                  const std::vector<memory_requirement*>& memory_requirements);

  kernel_launcher& get_launcher();
  const kernel_launcher& get_launcher() const;

  const std::vector<memory_requirement*>& get_memory_requirements() const;

private:
  kernel_launcher _launcher;
  std::vector<memory_requirement*> _memory_requirements;
};


/// An explicit memory operation
class memcpy_operation : public operation
{
public:
  virtual ~memcpy_operation(){}
  virtual std::size_t get_num_dimensions() const = 0;
  virtual std::size_t get_transferred_size() const = 0;
  virtual device_id get_source_device() const = 0;
  virtual device_id get_dest_device() const = 0;
  virtual void* get_dest_memory() const = 0;
  virtual const void* get_source_memory() const = 0;

  bool is_memcpy1d() const { return get_num_dimensions() == 1; }
  bool is_memcpy2d() const { return get_num_dimensions() == 2; }
  bool is_memcpy3d() const { return get_num_dimensions() == 3; }
};

class memcpy_operation1d : public memcpy_operation
{
public:
  memcpy_operation1d(
    device_id source_device, 
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t element_size,
    sycl::range<1> num_elements);

  std::size_t get_num_dimensions() const override;
  std::size_t get_transferred_size() const override;
  device_id get_source_device() const override;
  device_id get_dest_device() const override;
  void* get_dest_memory() const override;
  const void* get_source_memory() const override;

private:
  device_id _source_dev;
  device_id _dest_dev;
  const void* _src;
  void* _dest;
  std::size_t _element_size;
  sycl::range<1> _count;
};

class memcpy_operation2d : public memcpy_operation
{
public:
  memcpy_operation2d(
    device_id source_device,
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t src_pitch_in_bytes,
    std::size_t dest_pitch_in_bytes,
    std::size_t element_size,
    sycl::range<2> num_elements);

  std::size_t get_num_dimensions() const override;
  std::size_t get_transferred_size() const override;
  device_id get_source_device() const override;
  device_id get_dest_device() const override;
  void* get_dest_memory() const override;
  const void* get_source_memory() const override;

private:
  device_id _source_dev;
  device_id _dest_dev;
  const void* _src;
  void* _dest;
  std::size_t _src_pitch;
  std::size_t _dest_pitch;
  std::size_t _element_size;
  sycl::range<2> _count;
};

class memcpy_operation3d : public memcpy_operation
{
public:
  memcpy_operation3d(
    device_id source_device,
    device_id dest_device,
    const void* src_ptr,
    void* dest_ptr,
    std::size_t element_size,
    sycl::range<3> source_buffer_range,
    sycl::range<3> dest_buffer_range,
    sycl::range<3> num_elements);

  std::size_t get_num_dimensions() const override;
  std::size_t get_transferred_size() const override;
  device_id get_source_device() const override;
  device_id get_dest_device() const override;
  void* get_dest_memory() const override;
  const void* get_source_memory() const override;

private:
  device_id _source_dev;
  device_id _dest_dev;
  const void* _src;
  void* _dest;
  std::size_t _element_size;
  sycl::range<3> _source_buffer_range;
  sycl::range<3> _dest_buffer_range;
  sycl::range<3> _count;
};

/// A prefetch operation on SVM/USM memory
class prefetch_operation : public operation
{
  // TBD
};

template<class T, typename... Args>
std::unique_ptr<operation> make_operation(Args... args)
{
  return std::make_unique<T>(args...);
}

class requirements_list
{
public:
  template<class T, typename... Args>
  void add_requirement(Args... args)
  {
    std::unique_ptr<requirement> req = std::make_unique<T>(args...);
    this->add_requirement(std::move(req));
  }

  void add_requirement(std::unique_ptr<requirement> req)
  {
    auto node = std::make_shared<dag_node>(
      execution_hints{}, 
      std::vector<dag_node_ptr>{},
      std::move(req));
    
    _reqs.push_back(node);
  }

  const std::vector<dag_node_ptr>& get() const
  { return _reqs; }
  
private:
  std::vector<dag_node_ptr> _reqs;
};

}
}
}

#endif
