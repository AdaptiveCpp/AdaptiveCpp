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

#include "data.hpp"
#include "event.hpp"
#include "device_id.hpp"

#include <cstring>
#include <functional>
#include <memory>

namespace cl {
namespace sycl {
namespace detail {

using cost_type = double;

template<class T> class data_region;
using buffer_data_region = data_region<void *>;

class backend_executor;
class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

class operation
{
public:
  virtual ~operation(){}

  virtual cost_type get_runtime_costs() { return 1.; }
  virtual bool is_requirement() const { return false; }
  virtual bool is_synchronization_op() const { return false;}
};


class requirement : public operation
{
public:
  virtual bool is_memory_requirement() const = 0;

  virtual bool is_requirement() const final override
  { return true; }

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
      _offset = sycl::id<3>{0, 0, offset[0]};
      _range = sycl::range<3>{1, 1, range[0]};
    }
    else if(Dim == 2){
      _offset = sycl::id<3>{0, offset[0], offset[1]};
      _range = sycl::range<3>{1, range[0], range[1]};
    }
    else {
      _offset = offset;
      _range = range;
    }

  }

  std::size_t get_required_size() const override
  {
    return _range.size() * get_element_size();
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

// To describe memcpy operations, we need an abstract
// representations of memory locations. This is because,
// due to lazy allocation, we may not have allocated
// the target memcpy location, so we cannot know it in general.
class memory_location
{
public:
  memory_location(device_id d, sycl::id<3> access_offset,
                  std::shared_ptr<buffer_data_region> data_region);


  memory_location(device_id d, void *base_ptr, sycl::id<3> access_offset,
                  sycl::range<3> allocation_shape, std::size_t element_size);

  device_id get_device() const;

  sycl::id<3> get_access_offset() const;
  sycl::range<3> get_allocation_shape() const;

  std::size_t get_element_size() const;

  bool has_data_region() const;
  bool has_raw_pointer() const;

  // Note: Before materializing allocations after completion of the
  // dag_expander phase, the pointers returned here will be invalid
  // if the object was constructed from a data_region!
  void* get_base_ptr() const;
  void* get_access_ptr() const;
private:
  device_id _dev;

  sycl::id<3> _offset;
  sycl::range<3> _allocation_shape;
  std::size_t _element_size;

  bool _has_data_region;

  // Only valid if constructed with raw pointer constructor
  void *_raw_data;
  // Only valid if constructed data_region constructor
  std::shared_ptr<buffer_data_region> _data_region;
};

/// An explicit memory operation
class memcpy_operation : public operation
{
public:
  memcpy_operation(const memory_location &source, const memory_location &dest,
                   sycl::range<3> num_source_elements);
  
  
  std::size_t get_num_transferred_bytes() const;
  const memory_location &source() const;
  const memory_location &dest() const;

private:
  memory_location _source;
  memory_location _dest;
  sycl::range<3> _num_elements;
};

/// A prefetch operation on SVM/USM memory
class prefetch_operation : public operation
{
  // TBD
};


class backend_synchronization_operation : public operation
{
public:
  virtual ~backend_synchronization_operation(){}

  virtual bool is_synchronization_op() const override { return true; }
};

class event_before_node : public backend_synchronization_operation
{
public:
  event_before_node();

private:
  std::unique_ptr<dag_node_event> _evt;
  
};

class event_after_node : public backend_synchronization_operation
{
public:
  event_after_node(dag_node_ptr node);
};

class wait_for_node_on_same_lane : public backend_synchronization_operation
{
public:
  wait_for_node_on_same_lane(dag_node_ptr node);
};

class wait_for_node_on_same_backend : public backend_synchronization_operation
{
public:
  wait_for_node_on_same_backend(dag_node_ptr node);
};

class wait_for_external_node : public backend_synchronization_operation
{
public:
  wait_for_external_node(dag_node_ptr node);
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

  void add_requirement(std::unique_ptr<requirement> req);

  const std::vector<dag_node_ptr>& get() const;
  
private:
  std::vector<dag_node_ptr> _reqs;
};


}
}
}

#endif
