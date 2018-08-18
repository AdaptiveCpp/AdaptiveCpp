/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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


#ifndef SYCU_ACCESSOR_HPP
#define SYCU_ACCESSOR_HPP

#include <type_traits>
#include "range.hpp"
#include "access.hpp"
#include "item.hpp"
#include "buffer_allocator.hpp"
#include "backend/backend.hpp"


namespace cl {
namespace sycl {


template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator<T>>
class buffer;

class handler;

namespace detail {
namespace buffer {

template<class Buffer_type>
void* access_host_ptr(Buffer_type& b, access::mode m, hipStream_t stream);

template<class Buffer_type>
void* access_device_ptr(Buffer_type& b, access::mode m, hipStream_t stream);

template<class Buffer_type>
sycl::range<Buffer_type::buffer_dim> get_buffer_range(const Buffer_type& b);
} // buffer

namespace handler {
hipStream_t get_handler_stream(const sycl::handler& h);
} // handler

namespace accessor {

template<class T, access::mode m>
struct accessor_pointer_type
{
  using value = T*;
};

template<class T>
struct accessor_pointer_type<T, access::mode::read>
{
  using value = const T*;
};

} // accessor
} // detail

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget = access::target::global_buffer,
         access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor
{
public:
  using value_type = dataT;
  using reference = dataT &;
  using const_reference = const dataT &;

  using pointer_type =
    typename detail::accessor::accessor_pointer_type<dataT, accessmode>::value;

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions == 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                             T == access::target::host_buffer) ||
                            (P == access::placeholder::true_t  &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                             D == 0 >* = nullptr>
  accessor(buffer<dataT, 1> &bufferRef);

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions == 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer )) &&
                             D == 0 >* = nullptr>
  accessor(buffer<dataT, 1> &bufferRef, handler &commandGroupHandlerRef);

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions > 0 */

  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                             T == access::target::host_buffer) ||
                            (P == access::placeholder::true_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                            (D > 0)>* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef)
  {
    // ToDo think about when we need to update device/host buffers
    if(accessTarget == access::target::host_buffer)
    {
      this->_ptr = reinterpret_cast<pointer_type>(detail::buffer::access_host_ptr(
                                              bufferRef,
                                              accessmode,
                                              0));
      this->_range = detail::buffer::get_buffer_range(bufferRef);
    }
    else
    {
      // ToDo
      throw unimplemented{"accessor() with placeholder is unimplemented"};
    }
  }

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions > 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                            (D > 0)>* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef,
           handler &commandGroupHandlerRef)
  {
    hipStream_t stream =
        detail::handler::get_handler_stream(commandGroupHandlerRef);

    this->_ptr = reinterpret_cast<pointer_type>(
      detail::buffer::access_device_ptr(bufferRef,
                                        accessmode,
                                        stream));
    this->_range = detail::buffer::get_buffer_range(bufferRef);
  }

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions > 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                             T == access::target::host_buffer) ||
                            (P == access::placeholder::true_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                            (D > 0) >* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef,
           range<dimensions> accessRange,
           id<dimensions> accessOffset = {})
  {
    this->_ptr = reinterpret_cast<pointer_type>(
      detail::buffer::access_device_ptr(bufferRef,
                                        accessmode,
                                        0));
    this->_range = detail::buffer::get_buffer_range(bufferRef);
  }

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions > 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                            (D > 0)>* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset = {})
  {
    hipStream_t stream =
        detail::handler::get_handler_stream(commandGroupHandlerRef);

    this->_ptr = reinterpret_cast<pointer_type>(
      detail::buffer::access_device_ptr(bufferRef,
                                        accessmode,
                                        stream));
    this->_range = detail::buffer::get_buffer_range(bufferRef);
  }

  /* -- common interface members -- */
  __host__ __device__
  constexpr bool is_placeholder() const
  {
    return isPlaceholder == access::placeholder::true_t;
  }

  __host__ __device__
  size_t get_size() const
  {
    return get_count() * sizeof(dataT);
  }

  template<int D = dimensions,
           typename = std::enable_if_t<(D > 0)>>
  __host__ __device__
  size_t get_count() const
  {
    return _range.size();
  }

  template<int D = dimensions,
           std::enable_if_t<D == 0>* = nullptr>
  __host__ __device__
  size_t get_count() const
  { return 1; }

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           std::enable_if_t<(D > 0)>* = nullptr>
  __host__ __device__
  range<dimensions> get_range() const
  {
    return _range;
  }

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           typename = std::enable_if_t<(D > 0)>>
  __host__ __device__
  range<dimensions> get_offset() const
  {
    // ToDo: Properly implement access offsets
    return range<dimensions>{};
  }
  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions == 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           typename = std::enable_if_t<(M == access::mode::write ||
                                        M == access::mode::read_write ||
                                        M == access::mode::discard_write ||
                                        M == access::mode::discard_read_write) &&
                                       (D == 0)>>
  __host__ __device__
  operator dataT &() const
  {
    return *_ptr;
  }

  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions > 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           typename = std::enable_if_t<(M == access::mode::write ||
                                        M == access::mode::read_write ||
                                        M == access::mode::discard_write ||
                                        M == access::mode::discard_read_write) &&
                                       (D > 0)>>
  __host__ __device__
  dataT &operator[](id<dimensions> index) const
  {

    return _ptr[detail::linear_id<dimensions>::get(index, _range)];
  }

  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions == 1) */

  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(M == access::mode::write ||
                                        M == access::mode::read_write ||
                                        M == access::mode::discard_write ||
                                        M == access::mode::discard_read_write)
                                    && (D == 1)>>
  __host__ __device__
  dataT &operator[](size_t index) const
  {
    return _ptr[index];
  }


  /* Available only when: accessMode == access::mode::read && dimensions == 0 */
  template<access::mode M = accessmode,
           int D = dimensions,
           typename = std::enable_if_t<M == access::mode::read && D == 0>>
  __host__ __device__
  operator dataT() const
  {
    return *_ptr;
  }

  /* Available only when: accessMode == access::mode::read && dimensions > 0 */
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D > 0) && (M == access::mode::read)>>
  __host__ __device__
  dataT operator[](id<dimensions> index) const
  { return _ptr[detail::linear_id<dimensions>::get(index, _range)]; }

  /* Available only when: accessMode == access::mode::read && dimensions == 1 */
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D == 1) && (M == access::mode::read)>>
  __host__ __device__
  dataT operator[](size_t index) const
  { return _ptr[index]; }


  /* Available only when: accessMode == access::mode::atomic && dimensions == 0*/
  //operator atomic<dataT, access::address_space::global_space> () const;

  /* Available only when: accessMode == access::mode::atomic && dimensions > 0*/
  //atomic<dataT, access::address_space::global_space> operator[](
  //    id<dimensions> index) const;

  //atomic<dataT, access::address_space::global_space> operator[](
  //    size_t index) const;

  /* Available only when: dimensions > 1 */
  template<int D = dimensions,
           typename = std::enable_if_t<(D > 1)>>
  __host__ __device__
  accessor<dataT, dimensions-1, accessmode, accessTarget, isPlaceholder>
  operator[](size_t index) const
  {
    // Create sub accessor for slice at _ptr[index][...]
    accessor<dataT, dimensions-1, accessmode, accessTarget, isPlaceholder> sub_accessor;
    sub_accessor._range = detail::range::omit_first_dimension(this->_range);
    sub_accessor._ptr = this->_ptr + index * sub_accessor._range.size();

    return sub_accessor;
  }

  /* Available only when: accessTarget == access::target::host_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T==access::target::host_buffer>>
  dataT *get_pointer() const
  {
    return const_cast<dataT*>(this->_ptr);
  }

  /* Available only when: accessTarget == access::target::global_buffer */
  //global_ptr<dataT> get_pointer() const;

  /* Available only when: accessTarget == access::target::constant_buffer */
  //constant_ptr<dataT> get_pointer() const;
private:

  __host__ __device__
  accessor(){}

  range<dimensions> _range;
  pointer_type _ptr;
};

namespace detail {
namespace accessor {


template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
__host__ __device__
static
dataT* get_accessor_ptr(const sycl::accessor<dataT,
                                             dimensions,
                                             accessmode,
                                             accessTarget,
                                             isPlaceholder>& a)
{
  return a.get_pointer();
}


} // accessor
} // detail


} // sycl
} // cl

#endif
