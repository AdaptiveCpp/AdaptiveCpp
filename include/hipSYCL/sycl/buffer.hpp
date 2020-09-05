/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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


#ifndef HIPSYCL_BUFFER_HPP
#define HIPSYCL_BUFFER_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <type_traits>
#include <algorithm>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "property.hpp"
#include "types.hpp"
#include "context.hpp"
#include "buffer_allocator.hpp"

#include "id.hpp"
#include "range.hpp"

#include "../common/reinterpret_pointer_cast.hpp"

#include "accessor.hpp"

namespace hipsycl {
namespace sycl {

namespace property::buffer {

class use_host_ptr : public detail::property
{
public:
  use_host_ptr() = default;
};

class use_mutex : public detail::property
{
public:
  use_mutex(mutex_class& ref);
  mutex_class* get_mutex_ptr() const;
};

class context_bound : public detail::property
{
public:
  context_bound(context bound_context)
    : _ctx{bound_context}
  {}

  context get_context() const
  {
    return _ctx;
  }
private:
  context _ctx;
};

class use_optimized_host_memory : public detail::property
{};

} // property::buffer

namespace detail::buffer_policy {

class destructor_waits : public property
{ 
public: 
  destructor_waits(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};

class writes_back : public property
{ 
public: 
  writes_back(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};

class use_external_storage : public property
{ 
public: 
  use_external_storage(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};


}

namespace detail {

struct buffer_impl
{
  std::mutex lock;
  // Only used if a shared_ptr is passed to set_final_data()
  std::shared_ptr<void> writeback_buffer;
  // Only used if writeback is enabled
  void* writeback_ptr;
  // Only used if a shared_ptr is passed to the buffer constructor
  std::shared_ptr<void> shared_host_data;
  
  std::shared_ptr<rt::buffer_data_region> data;

  bool writes_back;
  bool destructor_waits;
  bool use_external_storage;

  ~buffer_impl() {
    if (writes_back) {
      if (!writeback_ptr) {
        HIPSYCL_DEBUG_WARNING
            << "buffer_impl::~buffer_impl: Writeback was requested but "
               "writeback pointer is null. Skipping write-back."
            << std::endl;
      } else {
        HIPSYCL_DEBUG_INFO
            << "buffer_impl::~buffer_impl: Preparing submission of writeback..."
            << std::endl;
        
        rt::dag_build_guard build{rt::application::dag()};

        auto explicit_requirement =
            rt::make_operation<rt::buffer_memory_requirement>(
                data, rt::id<3>{}, data->get_num_elements(),
                sycl::access::mode::read, sycl::access::target::host_buffer);

        rt::execution_hints enforce_bind_to_host;
        enforce_bind_to_host.add_hint(
            rt::make_execution_hint<rt::hints::bind_to_device>(
                detail::get_host_device()));

        build.builder()->add_explicit_mem_requirement(
            std::move(explicit_requirement), rt::requirements_list{},
            enforce_bind_to_host);

        // TODO what about writeback to external location set with
        // set_final_data()? -> need to submit an explicit copy
        // TODO Accessing the data allocations directly here is racy
        // since they might be modified during a DAG flush operation
        // simultaneously
        if(data->has_allocation(get_host_device())){
          if(data->get_memory(get_host_device()) != this->writeback_ptr){
            assert(false && "Writing back to external locations (not passed to "
                            "buffer at construction) is unimplemented");
          }
        }
      }
    }
    if(destructor_waits) {
      HIPSYCL_DEBUG_INFO
          << "buffer_impl::~buffer_impl: Waiting for operations to complete..."
          << std::endl;

      auto buffer_users = data->get_users().get_users();
      for(auto& user : buffer_users) {
        // This should not happen, as the scheduler registers users
        // after submitting them
        if(!user.user->is_submitted()) {
          HIPSYCL_DEBUG_WARNING
              << "buffer_impl::~buffer_impl: dag node is registered as user "
                 "but not marked as submitted, performing emergency DAG flush."
              << std::endl;

          rt::application::dag().flush_sync();
        }
        assert(user.user->is_submitted());
        user.user->wait();
      }
    }
  }
};

}


// Default template arguments for the buffer class
// are defined when forward-declaring the buffer in accessor.hpp
template <typename T, int dimensions,
          typename AllocatorT>
class buffer : public detail::property_carrying_object
{
public:
  template <class OtherT, int OtherDim, typename OtherAllocator>
  friend class buffer;

  template<class AccessorType, class BufferType, int Dim>
  friend void detail::accessor::bind_to_buffer(
    AccessorType& acc, BufferType& buff, 
    sycl::id<Dim> accessOffset, sycl::range<Dim> accessRange);
  
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  static constexpr int buffer_dim = dimensions;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    
    init_policies_from_properties_or_default(dpol);

    this->init(bufferRange);

    if(_impl->use_external_storage) {
      HIPSYCL_DEBUG_WARNING
          << "buffer: was constructed with use_external_storage but no host "
             "pointer was supplied. Cannot initialize this buffer with "
             "external storage."
          << std::endl;
    }
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : buffer(bufferRange, propList)
  { _alloc = allocator; }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;
    dpol.writes_back = true;
    
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage)
      this->init(bufferRange, hostData);
    else {
      this->init(bufferRange);
      copy_host_content(hostData);
    }
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
    : buffer{hostData, bufferRange, propList}
  {
    _alloc = allocator;
  }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    if(!_impl->use_external_storage) {
      // Construct buffer
      this->init(bufferRange);
      // Only use hostData for initialization
      copy_host_content(hostData);
    } else {
      HIPSYCL_DEBUG_WARNING
          << "buffer: constructed with property use_external_storage, but user "
             "passed a const pointer to buffer constructor. Removing const to enforce "
             "requested view semantics."
          << std::endl;
      this->init(bufferRange, const_cast<T*>(hostData));
    }
  }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
    : buffer{hostData, bufferRange, propList}
  { _alloc = allocator; }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();
    _alloc = allocator;

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;
    dpol.writes_back = true;
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage) {
      _impl->shared_host_data = hostData;
      this->init(bufferRange, hostData.get());
    } else {
      this->init(bufferRange);
      this->copy_host_content(hostData.get());
    }
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
  : buffer(hostData, bufferRange, AllocatorT(), propList)
  {}

  template <class InputIterator,
            int D = dimensions,
            typename = std::enable_if_t<D==1>>
  buffer(InputIterator first, InputIterator last,
         AllocatorT allocator,
         const property_list &propList = {})
  : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage)
      // TODO This could be allowed for special cases, e.g. if iterators are pointers
      throw invalid_parameter_error{
          "buffer: Cannot comply: User requested to using external storage, "
          "but this is not yet possible with iterators."};

    _alloc = allocator;

    std::size_t num_elements = std::distance(first, last);
    std::vector<T> contiguous_buffer(num_elements);
    std::copy(first, last, contiguous_buffer.begin());

    // Construct buffer
    this->init(range<1>{num_elements});

    // Only use hostData for initialization
    copy_host_content(contiguous_buffer.data());
  }

  template <class InputIterator, int D = dimensions,
            typename = std::enable_if_t<D == 1>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
  : buffer(first, last, AllocatorT(), propList) 
  {}

  buffer(buffer<T, dimensions, AllocatorT> b,
         const id<dimensions> &baseIndex,
         const range<dimensions> &subRange)
  {
    assert(false && "subbuffer is unimplemented");
  }

  range<dimensions> get_range() const
  {
    return _range;
  }

  std::size_t get_count() const
  {
    return _range.size();
  }

  std::size_t get_size() const
  {
    return get_count() * sizeof(T);
  }

  AllocatorT get_allocator() const
  {
    return _alloc;
  }

  template <access::mode mode, access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target> get_access(handler &commandGroupHandler)
  {
    return accessor<T, dimensions, mode, target>{*this, commandGroupHandler};
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer> get_access()
  {
    return accessor<T, dimensions, mode, access::target::host_buffer>{*this};
  }

  template <access::mode mode, access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target> get_access(
      handler &commandGroupHandler, range<dimensions> accessRange,
      id<dimensions> accessOffset = {})
  {
    return accessor<T, dimensions, mode, target>{
      *this, commandGroupHandler, accessRange, accessOffset
    };
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer> get_access(
      range<dimensions> accessRange, id<dimensions> accessOffset = {})
  {
    return accessor<T, dimensions, mode, access::target::host_buffer>{
      *this, accessRange, accessOffset
    };
  }

  void set_final_data(std::shared_ptr<T> finalData)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    set_write_back_target(finalData.get());
    
    _impl->writeback_buffer = finalData;
  }

  // TODO Add special handling of iterators for set_final_data()
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    set_write_back_target(finalData);
  }

  void set_write_back(bool flag = true)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    this->enable_write_back(flag);
  }

  // ToDo Subbuffers are unsupported
  bool is_sub_buffer() const
  { return false; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {

    buffer<ReinterpretT, ReinterpretDim> new_buffer;
    new_buffer.init_from(*this);
    new_buffer._range = reinterpretRange;
    
    return new_buffer;
  }

  friend bool operator==(const buffer& lhs, const buffer& rhs)
  {
    return lhs._impl == rhs._impl;
  }

  friend bool operator!=(const buffer& lhs, const buffer& rhs)
  {
    return !(lhs == rhs);
  }

private:
  struct default_policies
  {
    bool destructor_waits;
    bool writes_back;
    bool use_external_storage; 
  };

  template <typename Destination = std::nullptr_t>
  void set_write_back_target(Destination finalData = nullptr)
  {
    if (finalData != nullptr) {
      enable_write_back(true);
    }
    else {
      enable_write_back(false);
    }
    
    _impl->writeback_ptr = finalData;
  }

  void enable_write_back(bool flag) {

    if(this->has_property<detail::buffer_policy::writes_back>()){
      if (_impl->writes_back != flag){
        // Deny changing policy if it has previously been explicitly requested
        // by the user
        throw invalid_parameter_error{
            "buffer::set_write_back(): buffer was constructed explicitly with "
            "writeback policy, denying changing the policy as this likely "
            "indicates a bug in user code"};
      }
    }
    if(_impl->writes_back != flag) {
      HIPSYCL_DEBUG_INFO << "buffer: Changing write back policy to: " << flag
                         << std::endl;
      _impl->writes_back = flag;
    }
  }

  void copy_host_content(const T* data)
  {
    assert(_impl);
    auto host_device = detail::get_host_device();
    preallocate_host_buffer();
    std::memcpy(_impl->data->get_memory(host_device), data,
                sizeof(T) * _range.size());
  }

  void init_policies_from_properties_or_default(default_policies dpol)
  {
    _impl->destructor_waits = get_policy_from_property_or_default<
        detail::buffer_policy::destructor_waits>(dpol.destructor_waits);
    
    _impl->writes_back =
        get_policy_from_property_or_default<detail::buffer_policy::writes_back>(
            dpol.writes_back);

    _impl->use_external_storage = get_policy_from_property_or_default<
        detail::buffer_policy::use_external_storage>(dpol.use_external_storage);
  }

  template<class Policy>
  bool get_policy_from_property_or_default(bool default_value)
  {
    if(this->has_property<Policy>())
      return this->get_property<Policy>().value();
    return default_value;
  }

  
  buffer()
  : detail::property_carrying_object {property_list {}}
  {}

  template <class OtherT, int OtherDim>
  void init_from(const buffer<OtherT, OtherDim> &other) {
    detail::property_carrying_object::operator=(other);
    this->_alloc = other._alloc;
    this->_range = other._range;
    this->_impl = other._impl;
  }
  
  void init_data_backend(const range<dimensions>& range)
  {
    this->_range = range;
    // TODO properly set page size and expose configurable page size
    // to user
    rt::range<3> page_size = rt::embed_in_range3(range);

    auto on_destruction = [](rt::buffer_data_region* data) {};

    _impl->data = std::make_shared<rt::buffer_data_region>(
        rt::embed_in_range3(range), sizeof(T), page_size,
        on_destruction);
  }

  void preallocate_host_buffer()
  {
    void* host_ptr = nullptr;
    rt::device_id host_device = detail::get_host_device();

    if(!_impl->data->has_allocation(host_device)){
      if(this->has_property<property::buffer::use_optimized_host_memory>()){
        // TODO: Actually may need to use non-host backend here...
        host_ptr =
            rt::application::get_backend(host_device.get_backend())
                .get_allocator(host_device)
                ->allocate_optimized_host(
                    128, _impl->data->get_num_elements().size() * sizeof(T));
      } else {
        host_ptr =
            rt::application::get_backend(host_device.get_backend())
                .get_allocator(host_device)
                ->allocate(
                    128, _impl->data->get_num_elements().size() * sizeof(T));
      }

      if(!host_ptr)
        throw runtime_error{"buffer: host memory allocation failed"};

      
      _impl->data->add_empty_allocation(host_device, host_ptr,
                                        true /*takes_ownership*/);
    }
  }

  void init(const range<dimensions>& range)
  {

    this->init_data_backend(range);
    // necessary to preallocate to make sure potential optimized memory
    // can be allocated
    preallocate_host_buffer();
  }

  void init(const range<dimensions>& range, T* host_memory)
  {
    if(!host_memory)
      throw invalid_parameter_error{"buffer: Supplied host pointer is null."};

    if(this->has_property<property::buffer::use_optimized_host_memory>()){
      HIPSYCL_DEBUG_INFO
          << "buffer: was constructed with use_optimized_host_memory property, "
             "but also as view for existing host memory. "
             "use_optimized_host_memory will have no effect, since "
             "the buffer will rely on existing memory instead of allocating itself."
          << std::endl;
    }

    this->init_data_backend(range);

    _impl->data->add_nonempty_allocation(detail::get_host_device(), host_memory,
                                         false /*takes_ownership*/);
    // Remember host_memory in case of potential write back
    _impl->writeback_ptr = host_memory;
  }


  AllocatorT _alloc;
  range<dimensions> _range;

  std::shared_ptr<detail::buffer_impl> _impl;
};

namespace detail::accessor {

template<class AccessorType, class BufferType, int Dim>
void bind_to_buffer(AccessorType& acc, BufferType& buff, 
    sycl::id<Dim> accessOffset, sycl::range<Dim> accessRange) {
  
  acc._offset = accessOffset;
  acc._range = accessRange;
  acc._buffer_range = buff.get_range();
  acc.set_data_region(buff._impl->data);
}

template<class AccessorType, class T, int Dim, class AllocatorT>
void bind_to_buffer_with_defaults(AccessorType& acc, buffer<T,Dim,AllocatorT>& buff)
{
  bind_to_buffer(acc, buff, sycl::id<Dim>{}, buff.get_range());
}

template<class AccessorType, class BufferType>
void bind_to_buffer(AccessorType& acc, BufferType& buff) {
  bind_to_buffer_with_defaults(acc, buff);
}

}

} // sycl
} // hipsycl

#endif
