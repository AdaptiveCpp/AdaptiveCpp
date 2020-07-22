/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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

#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/range.hpp"
#include "hipSYCL/sycl/access.hpp"

#include "hipSYCL/glue/deferred_pointer.hpp"

#include "data.hpp"
#include "event.hpp"
#include "device_id.hpp"
#include "kernel_launcher.hpp"
#include "util.hpp"
#include "hw_model/cost.hpp"

#include <cstring>
#include <functional>
#include <memory>

namespace hipsycl {
namespace rt {

class inorder_queue;

template<class T> class data_region;
using buffer_data_region = data_region<void *>;

class backend_executor;
class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

class kernel_operation;
class memcpy_operation;
class prefetch_operation;

class operation_dispatcher
{
public:
  virtual void dispatch_kernel(kernel_operation* op) = 0;
  virtual void dispatch_memcpy(memcpy_operation* op) = 0;
  virtual void dispatch_prefetch(prefetch_operation* op) = 0;
  virtual ~operation_dispatcher(){}
};

class operation
{
public:
  virtual ~operation(){}

  virtual cost_type get_runtime_costs() { return 1.; }
  virtual bool is_requirement() const { return false; }
  virtual bool is_data_transfer() const { return false; }

  virtual void dispatch(operation_dispatcher* dispatch) = 0;
};


class requirement : public operation
{
public:
  virtual bool is_memory_requirement() const = 0;

  virtual bool is_requirement() const final override
  { return true; }

  virtual void dispatch(operation_dispatcher* dispatch) final override
  { assert(false && "Cannot dispatch implicit requirements"); }

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

  virtual bool intersects_with(const memory_requirement* other) const = 0;
  /// Check if this requirement's data region intersects with an existing data usage.
  /// Note: Assumes that the data usage operations on the same memory object!
  virtual bool intersects_with(const data_user& user) const = 0;

  bool is_buffer_requirement() const 
  { return !is_image_requirement(); }
};


class buffer_memory_requirement : public memory_requirement
{
public:
  template <int Dim>
  buffer_memory_requirement(std::shared_ptr<buffer_data_region> mem_region,
                            const sycl::id<Dim> offset,
                            const sycl::range<Dim> range,
                            sycl::access::mode access_mode,
                            sycl::access::target access_target)
      : _mem_region{mem_region}, _element_size{mem_region->get_element_size()},
        _mode{access_mode}, _target{access_target}, _dimensions{Dim},
        _device_data_location{nullptr}
  {
    static_assert(Dim >= 1 && Dim <=3, 
      "dimension of buffer memory requirement must be between 1 and 3");

    _offset = embed_in_id3(offset);
    _range = embed_in_range3(range);
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

  bool intersects_with(const data_user& user) const override {
    auto other_page_range =
        _mem_region->get_page_range(user.offset, user.range);

    return page_ranges_intersect(other_page_range);
  }

  bool intersects_with(const memory_requirement* other) const override {

    if(!other->is_buffer_requirement())
      return false;

    const buffer_memory_requirement *other_buff =
        cast<const buffer_memory_requirement>(other);

    if(_mem_region != other_buff->_mem_region)
      return false;

    auto other_page_range = _mem_region->get_page_range(
        other_buff->get_access_offset3d(), other_buff->get_access_range3d());

    
    return page_ranges_intersect(other_page_range);
  }

  bool has_device_data_location() const {
    return _device_data_location != nullptr;
  }

  void initialize_device_data(void *location) {
    assert(!has_device_data_location());
    _device_data_location = location;
  }

  template <class T> glue::deferred_pointer<T> make_deferred_pointer() const {
    return glue::deferred_pointer<T>{const_cast<void**>(&_device_data_location)};
  }
  
private:
  bool page_ranges_intersect(buffer_data_region::page_range other) const{
    auto my_range = _mem_region->get_page_range(_offset, _range);

    for(std::size_t dim = 0; dim < 3; ++dim) {
      auto begin1 = my_range.first[dim];
      auto end1 = begin1+my_range.second[dim];

      auto begin2 = other.first[dim];
      auto end2 = begin2+other.second[dim];
      // if at least one dimension does not intersect,
      // there is no intersection
      if(!(begin1 < end2 && begin2 < end1))
        return false;
    }

    return true;
  }

  std::shared_ptr<buffer_data_region> _mem_region;
  sycl::id<3> _offset;
  sycl::range<3> _range;
  std::size_t _element_size;
  sycl::access::mode _mode;
  sycl::access::target _target;
  int _dimensions;

  void* _device_data_location;
};


class requirements_list;

class kernel_operation : public operation
{
public:
  kernel_operation(const std::string& kernel_name,
                  std::vector<std::unique_ptr<backend_kernel_launcher>> kernels,
                  const requirements_list& requirements);

  kernel_launcher& get_launcher();
  const kernel_launcher& get_launcher() const;

  const std::vector<memory_requirement*>& get_requirements() const;

  void dispatch(operation_dispatcher* dispatcher) override {
    dispatcher->dispatch_kernel(this);
  }

private:
  std::string _kernel_name;
  kernel_launcher _launcher;
  std::vector<memory_requirement*> _requirements;
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
  // Only valid if constructed with data_region constructor
  std::shared_ptr<buffer_data_region> _data_region;
};

/// An explicit memory operation
class memcpy_operation : public operation
{
public:
  memcpy_operation(const memory_location &source, const memory_location &dest,
                   sycl::range<3> num_source_elements);


  std::size_t get_num_transferred_bytes() const;
  sycl::range<3> get_num_transferred_elements() const;
  
  const memory_location &source() const;
  const memory_location &dest() const;

  virtual bool is_data_transfer() const final override;
  virtual void dispatch(operation_dispatcher* op) final override {
    op->dispatch_memcpy(this);
  }
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


class backend_synchronization_operation
{
public:
  virtual ~backend_synchronization_operation(){}
  virtual cost_type get_runtime_costs() { return 1.; }

  virtual bool is_event_before_node() const { return false; }
  virtual bool is_event_after_node() const { return false; }
  virtual bool is_wait_operation() const { return false; }
};

class event_before_node : public backend_synchronization_operation
{
public:
  virtual ~event_before_node(){}

  virtual bool is_event_before_node() const final override { return true; }

  void assign_event(std::shared_ptr<dag_node_event> evt) {
    _evt = std::move(evt);
  }

  std::shared_ptr<dag_node_event> get() const { return _evt; }

private:
  std::shared_ptr<dag_node_event> _evt;
};

class event_after_node : public backend_synchronization_operation
{
public:
  virtual ~event_after_node(){}

  virtual bool is_event_after_node() const final override { return true; }

  void assign_event(std::shared_ptr<dag_node_event> evt) {
    _evt = std::move(evt);
  }

  std::shared_ptr<dag_node_event> get() const { return _evt; }
private:
  std::shared_ptr<dag_node_event> _evt;
};

enum class wait_target {
  same_lane,
  same_backend,
  external_backend
};

class wait_operation : public backend_synchronization_operation
{
public:
  wait_operation(dag_node_ptr target_node)
  : _target_node{target_node}
  {}

  virtual wait_target get_wait_target() const = 0;
  virtual bool is_wait_operation() const final override {return true;}

  virtual ~wait_operation(){}

  dag_node_ptr get_target_node() const
  { return _target_node; }

protected:
  dag_node_ptr _target_node;
};

class wait_for_node_on_same_lane : public wait_operation
{
public:
  wait_for_node_on_same_lane(dag_node_ptr node)
  : wait_operation{node} {}

  wait_target get_wait_target() const final override
  { return wait_target::same_lane; }
};

class wait_for_node_on_same_backend : public wait_operation
{
public:
  wait_for_node_on_same_backend(dag_node_ptr node)
  : wait_operation{node} {}

  wait_target get_wait_target() const final override
  { return wait_target::same_backend; }
};

class wait_for_external_node : public wait_operation
{
public:
  wait_for_external_node(dag_node_ptr node)
  : wait_operation{node} {}

  wait_target get_wait_target() const final override
  { return wait_target::external_backend; }
};



template<class T, typename... Args>
std::unique_ptr<operation> make_operation(Args... args)
{
  return std::make_unique<T>(std::forward<Args>(args)...);
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


#endif
