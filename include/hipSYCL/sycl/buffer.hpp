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
#include <type_traits>
#include <algorithm>

#include "property.hpp"
#include "types.hpp"
#include "context.hpp"
#include "buffer_allocator.hpp"

#include "id.hpp"
#include "range.hpp"

#include "detail/buffer.hpp"

#include "accessor.hpp"

namespace hipsycl {
namespace sycl {
namespace property {
namespace buffer {

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

} // property
} // buffer


// hipSYCL property extensions
namespace hipsycl {
namespace property {
namespace buffer {

struct use_svm : public detail::property
{};

struct try_pinned_memory : public detail::property
{};

}
}
}

// Default template arguments for the buffer class
// are defined when forward-declaring the buffer in accessor.hpp
template <typename T, int dimensions,
          typename AllocatorT>
class buffer : public detail::property_carrying_object
{
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  static constexpr int buffer_dim = dimensions;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
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
         const range<dimensions> &subRange);

  /* Available only when: dimensions == 1. */

  /* CL interop is not supported
  buffer(cl_mem clMemObject, const context &syclContext,
         event availableEvent = {});
  */

  /* -- common interface members -- */

  /* -- property interface members -- */

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

  void set_final_data(shared_ptr_class<T> finalData)
  {
    this->_writeback_buffer = finalData;
    this->set_final_data(finalData.get());
  }

  // TODO Add special handling of iterators for set_final_data()
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr)
  {
    this->_cleanup_trigger->remove_cleanup_callbacks();
    _buffer->set_write_back(finalData);
    if(finalData != nullptr)
      _buffer->enable_write_back(true);
  }

  void set_write_back(bool flag = true)
  {
    this->_cleanup_trigger->remove_cleanup_callbacks();
    this->_buffer->enable_write_back(flag);
  }

  // ToDo Implement
  bool is_sub_buffer() const;

  // ToDo Implement
  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT>
  reinterpret(range<ReinterpretDim> reinterpretRange) const;

  friend bool operator==(const buffer& lhs, const buffer& rhs)
  {
    return lhs->_buffer == rhs->_buffer;
  }

  friend bool operator!=(const buffer& lhs, const buffer& rhs)
  {
    return !(lhs == rhs);
  }

  detail::buffer_ptr _detail_get_buffer_ptr() const
  {
    return _buffer;
  }
private:
  void create_buffer(detail::device_alloc_mode device_mode,
                     detail::host_alloc_mode host_mode,
                     const range<dimensions>& range)
  {
#if !defined(HIPSYCL_CPU_EMULATE_SEPARATE_MEMORY) && defined(HIPSYCL_PLATFORM_CPU)
    // force svm allocation
    device_mode = detail::device_alloc_mode::svm;
    host_mode = detail::host_alloc_mode::svm;
#endif
    
    _buffer = detail::buffer_ptr{
      new detail::buffer_impl{
        sizeof(T) * range.size(), device_mode, host_mode
      }
    };
    _range = range;
  }

  void create_buffer(T* host_memory,
                     const range<dimensions>& range)
  {
#if !defined(HIPSYCL_CPU_EMULATE_SEPARATE_MEMORY) && defined(HIPSYCL_PLATFORM_CPU)
    bool force_svm = true;
#else
    bool force_svm = false;
#endif
      
    _buffer = detail::buffer_ptr{
      new detail::buffer_impl{
        sizeof(T) * range.size(),
        reinterpret_cast<void*>(host_memory),
        force_svm
      }
    };
    _range = range;
  }

  void init(const range<dimensions>& range)
  {

    detail::device_alloc_mode device_mode =
        detail::device_alloc_mode::regular;
    detail::host_alloc_mode host_mode =
        detail::host_alloc_mode::regular;

    if(this->has_property<hipsycl::property::buffer::try_pinned_memory>() &&
       this->has_property<hipsycl::property::buffer::use_svm>())
      throw invalid_parameter_error{"use_pinned_memory and use_svm "
                                    "as buffer properties cannot be specified "
                                    "at the same time."};

    if(this->has_property<hipsycl::property::buffer::try_pinned_memory>())
      host_mode = detail::host_alloc_mode::allow_pinned;

    
    if(this->has_property<hipsycl::property::buffer::use_svm>())
    {
      device_mode = detail::device_alloc_mode::svm;
      host_mode = detail::host_alloc_mode::svm;
    }

    this->create_buffer(device_mode,
                        host_mode,
                        range);
    this->_cleanup_trigger =
        std::make_shared<detail::buffer_cleanup_trigger>(_buffer);
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
  shared_ptr_class<T> _writeback_buffer;
  // Only used if a shared_ptr is passed to the buffer constructor
  shared_ptr_class<T> _shared_host_data;
  // Must be defined after the shared_ptrs to ensure that they still
  // exist once the cleanup trigger is executed!
  shared_ptr_class<detail::buffer_cleanup_trigger> _cleanup_trigger;
  
  detail::buffer_ptr _buffer;
};

namespace detail {
namespace buffer {


template<class Buffer_type>
buffer_ptr get_buffer_impl(Buffer_type& buff)
{
  return buff._detail_get_buffer_ptr();
}

template<class Buffer_type>
sycl::range<Buffer_type::buffer_dim> get_buffer_range(const Buffer_type& b)
{
  return b.get_range();
}

} // buffer
} // detail
} // sycl
} // hipsycl

#endif
