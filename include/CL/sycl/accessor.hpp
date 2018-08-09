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
#include "buffer_allocator.hpp"


namespace cl {
namespace sycl {


template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator<T>>
class buffer;

namespace detail {
namespace buffer {

template<class Buffer_type>
void* access_host_ptr(Buffer_type& b, access::mode m);

template<class Buffer_type>
void* access_device_ptr(Buffer_type& b, access::mode m);

}
}

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

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions == 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           typename = std::enable_if_t<(P == access::placeholder::false_t &&
                                        T == access::target::host_buffer) ||
                                       (P == access::placeholder::true_t  &&
                                       (T == access::target::global_buffer ||
                                        T == access::target::constant_buffer)) &&
                                        D == 0>>
  accessor(buffer<dataT, 1> &bufferRef);

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions == 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           typename = std::enable_if_t<(P == access::placeholder::false_t &&
                                       (T == access::target::global_buffer ||
                                        T == access::target::constant_buffer )) &&
                                        D == 0>>
  accessor(buffer<dataT, 1> &bufferRef, handler &commandGroupHandlerRef);

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions > 0 */

  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           typename = std::enable_if_t<(P == access::placeholder::false_t &&
                                        T == access::target::host_buffer) ||
                                       (P == access::placeholder::true_t &&
                                       (T == access::target::global_buffer ||
                                        T == access::target::constant_buffer)) &&
                                        D > 0>>
  accessor(buffer<dataT, dimensions> &bufferRef)
  {
    // ToDo think about when we need to update device/host buffers
    if(accessTarget == access::target::host_buffer)
      this->_ptr = reinterpret_cast<dataT*>(detail::buffer::access_host_ptr(
                                              bufferRef,
                                              accessmode));
    else
    {
      // ToDo
      this->_ptr = nullptr;
    }
  }

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions > 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           typename = std::enable_if_t<(P == access::placeholder::false_t &&
                                       (T == access::target::global_buffer ||
                                        T == access::target::constant_buffer)) &&
                                        D > 0>>
  accessor(buffer<dataT, dimensions> &bufferRef,
           handler &commandGroupHandlerRef)
  {
    this->_ptr = reinterpret_cast<dataT*>(detail::access_device_ptr(bufferRef,
                                            accessmode));
  }

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
accessTarget == access::target::host_buffer) ||
(isPlaceholder ==
access::placeholder::true_t && (accessTarget == access::target::global_buffer
|| accessTarget == access::target::constant_buffer)) && dimensions > 0 */
  accessor(buffer<dataT, dimensions> &bufferRef, range<dimensions> accessRange,
           id<dimensions> accessOffset = {});

  /* Available only when: (isPlaceholder == access::placeholder::false_t &&
(accessTarget == access::target::global_buffer || accessTarget ==
access::target::constant_buffer)) && dimensions > 0 */

  accessor(buffer<dataT, dimensions> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset = {});

  /* -- common interface members -- */
  __host__ __device__
  constexpr bool is_placeholder() const
  {
    return isPlaceholder == access::placeholder::true_t;
  }

  size_t get_size() const;

  size_t get_count() const;

  /* Available only when: dimensions > 0 */
  range<dimensions> get_range() const;

  /* Available only when: dimensions > 0 */
  range<dimensions> get_offset() const;
  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions == 0) */

  operator dataT &() const;

  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions > 0) */

  dataT &operator[](id<dimensions> index) const;

  /* Available only when: (accessMode == access::mode::write || accessMode ==
access::mode::read_write || accessMode == access::mode::discard_write ||
accessMode == access::mode::discard_read_write) && dimensions == 1) */
  dataT &operator[](size_t index) const;
  /* Available only when: accessMode == access::mode::read && dimensions == 0 */
  operator dataT() const;

  /* Available only when: accessMode == access::mode::read && dimensions > 0 */
  __host__ __device__
  dataT operator[](id<dimensions> index) const;
  /* Available only when: accessMode == access::mode::read && dimensions == 1 */
  __host__ __device__
  dataT operator[](size_t index) const;
  /* Available only when: accessMode == access::mode::atomic && dimensions == 0*/
  operator atomic<dataT, access::address_space::global_space> () const;
  /* Available only when: accessMode == access::mode::atomic && dimensions > 0*/
  atomic<dataT, access::address_space::global_space> operator[](
      id<dimensions> index) const;

  atomic<dataT, access::address_space::global_space> operator[](
      size_t index) const;
  /* Available only when: dimensions > 1 */
  __unspecified__ &operator[](size_t index) const;

  /* Available only when: accessTarget == access::target::host_buffer */
  dataT *get_pointer() const;

  /* Available only when: accessTarget == access::target::global_buffer */
  global_ptr<dataT> get_pointer() const;

  /* Available only when: accessTarget == access::target::constant_buffer */
  constant_ptr<dataT> get_pointer() const;
private:

  void init()

  range<dimensions> _range;
  dataT* _ptr;
};


}
}

#endif
