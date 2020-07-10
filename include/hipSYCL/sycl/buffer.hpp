/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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
#include <iterator>
#include <memory>
#include <type_traits>
#include <algorithm>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/util.hpp"
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
    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    this->init(bufferRange);
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : buffer(bufferRange, propList)
  {
    _alloc = allocator;
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;
    dpol.writes_back = true;
    init_policies_from_properties_or_default(dpol);

    this->init(bufferRange, hostData);
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
    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    // Construct buffer
    this->init(bufferRange);

    // Only use hostData for initialization
    _buffer->write(reinterpret_cast<const void*>(hostData), 0);
  }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
    : buffer{hostData, bufferRange, propList}
  {
    _alloc = allocator;
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : detail::property_carrying_object{propList},
      _shared_host_data{hostData}
  {
    _alloc = allocator;

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;
    dpol.writes_back = true;
    init_policies_from_properties_or_default(dpol);

    this->init(bufferRange, hostData.get());
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
    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    _alloc = allocator;

    constexpr bool is_const_iterator = 
        std::is_const<
          typename std::remove_reference<
            typename std::iterator_traits<InputIterator>::reference
          >::type
        >::value;

    std::size_t num_elements = std::distance(first, last);
    vector_class<T> contiguous_buffer(num_elements);
    std::copy(first, last, contiguous_buffer.begin());

    // Construct buffer
    this->init(range<1>{num_elements});

    // Only use hostData for initialization
    _buffer->write(reinterpret_cast<const void*>(contiguous_buffer.data()), 0);
    if(!is_const_iterator)
    {
      // If we are dealing with non-const iterators, we must also
      // writeback the results.
      // TODO: The spec seems to be contradictory if writebacks also
      // occur with iterators - investigate the desired behavior
      // TODO: If first, last are random access iterators, we can directly
      // the memory range [first, last) as writeback buffer
      this->_buffer->enable_write_back(true);

      
      auto buff = this->_buffer;
      this->_cleanup_trigger->add_cleanup_callback([first, last, buff](){
        const T* host_data = reinterpret_cast<T*>(buff->get_host_ptr());

        std::copy(host_data, host_data + std::distance(first,last), first);
      });
      
    }
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
    this->_writeback_buffer = finalData;
    this->set_final_data(finalData.get());
  }

  // TODO Add special handling of iterators for set_final_data()
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr)
  {
    if (finalData != nullptr) {
      set_write_back(true);
    }
    else {
      set_write_back(false);
    }
    _writeback_ptr = finalData;
  }

  void set_write_back(bool flag = true)
  {
    if(this->has_property<detail::buffer_policy::writes_back>()){
      if (_writes_back != flag){
        // Deny changing policy if it has previously been explicitly requested
        // by the user
        throw invalid_parameter_error{
            "buffer::set_write_back(): buffer was constructed explicitly with "
            "writeback policy, denying changing the policy as this likely "
            "indicates a bug in user code"};
      }
    }
    if(_writes_back != flag) {
      HIPSYCL_DEBUG_INFO << "buffer: Changing write back policy to: " << flag
                         << std::endl;
      _writes_back = flag;
    }
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

  bool operator==(const buffer& rhs) const
  {
    return _data == rhs->_data;
  }

  bool operator!=(const buffer& rhs) const
  {
    return !(*this == rhs);
  }

private:
  struct default_policies
  {
    bool destructor_waits;
    bool writes_back;
    bool use_external_storage; 
  };

  void init_policies_from_properties_or_default(default_policies dpol)
  {
    this->_destructor_waits = get_policy_from_property_or_default<
        detail::buffer_policy::destructor_waits>(dpol.destructor_waits);
    
    this->_writes_back =
        get_policy_from_property_or_default<detail::buffer_policy::writes_back>(
            dpol.writes_back);

    this->_use_external_storage = get_policy_from_property_or_default<
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
    this->_writeback_buffer = ::hipsycl::common::shim::reinterpret_pointer_cast<T>(other._writeback_buffer);
    this->_shared_host_data = ::hipsycl::common::shim::reinterpret_pointer_cast<T>(other._shared_host_data);
    this->_writeback_ptr = other._writeback_ptr;
    this->_data = other._data;

    this->_writes_back = other._writes_back;
    this->_destructor_waits = other._destructor_waits;
    this->_use_external_storage = other._use_external_storage;
  }
  
  void init(const range<dimensions>& range)
  {

    if(this->has_property<hipsycl::property::buffer::try_pinned_memory>() &&
       this->has_property<hipsycl::property::buffer::use_svm>())
      throw invalid_parameter_error{"use_pinned_memory and use_svm "
                                    "as buffer properties cannot be specified "
                                    "at the same time."};

    void* host_ptr = nullptr;
    rt::device_id host_device = detail::get_host_device();

    if(this->has_property<hipsycl::property::buffer::try_pinned_memory>()){
      host_ptr =
          rt::application::get_backend(host_device.get_backend())
              .get_allocator(host_device)
              ->allocate_transfer_optimized(128, range.size() * sizeof(T));
    }
    else if(this->has_property<hipsycl::property::buffer::use_svm>()){
      host_ptr = rt::application::get_backend(host_device.get_backend())
                     .get_allocator(host_device)
                     ->allocate_usm(range.size() * sizeof(T));
    }

    // TODO properly set page size and expose configurable page size
    // to user
    std::size_t page_size = range.size();
/*
data_region(sycl::range<3> num_elements, std::size_t element_size,
              std::size_t page_size, destruction_handler on_destruction)*/

    auto on_destruction = [](rt::buffer_data_region* data) {

    };

    this->_data = std::make_shared<rt::buffer_data_region>(
        rt::embed_in_range3(range), sizeof(T), page_size, on_destruction);
  }

  void init(const range<dimensions>& range, T* host_memory)
  {
    this->create_buffer(host_memory, range);
    this->_cleanup_trigger =
        std::make_shared<detail::buffer_cleanup_trigger>(_buffer);
  }


  AllocatorT _alloc;
  range<dimensions> _range;

  
  // Only used if a shared_ptr is passed to set_final_data()
  std::shared_ptr<T> _writeback_buffer;
  // Only used if writeback is enabled
  T* _writeback_ptr;
  // Only used if a shared_ptr is passed to the buffer constructor
  std::shared_ptr<T> _shared_host_data;
  
  std::shared_ptr<rt::buffer_data_region> _data;

  bool _writes_back;
  bool _destructor_waits;
  bool _use_external_storage;
};

namespace detail::accessor {

template<class AccessorType, class BufferType, int Dim>
void bind_to_buffer(AccessorType& acc, BufferType& buff, 
    sycl::id<Dim> accessOffset, sycl::range<Dim> accessRange) {
  
  acc._offset = accessOffset;
  acc._range = accessRange;
  acc._buffer_range = buff.get_range();
  acc.set_data_region(buff._data);
}

template<class AccessorType, class T, class Dim, class AllocatorT>
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
