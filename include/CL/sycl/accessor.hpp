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
#include "multi_ptr.hpp"
#include "detail/local_memory_allocator.hpp"
#include "detail/stream.hpp"
#include "detail/buffer.hpp"

namespace cl {
namespace sycl {


template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator<T>>
class buffer;

class handler;

namespace detail {
namespace buffer {

template<class Buffer_type>
buffer_ptr get_buffer_impl(Buffer_type& buff);

template<class Buffer_type>
sycl::range<Buffer_type::buffer_dim> get_buffer_range(const Buffer_type& b);
} // buffer

namespace handler {

template<class T>
detail::local_memory::address allocate_local_mem(cl::sycl::handler&,
                                                 size_t num_elements);

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

void* obtain_host_access(buffer_ptr buff,
                         access::mode access_mode);

void* obtain_device_access(buffer_ptr buff,
                           sycl::handler& cgh,
                           access::mode access_mode);

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
           std::enable_if_t<((P == access::placeholder::false_t &&
                              T == access::target::host_buffer) ||
                             (P == access::placeholder::true_t  &&
                             (T == access::target::global_buffer ||
                              T == access::target::constant_buffer))) &&
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
           std::enable_if_t<((P == access::placeholder::false_t &&
                              T == access::target::host_buffer) ||
                             (P == access::placeholder::true_t &&
                             (T == access::target::global_buffer ||
                              T == access::target::constant_buffer))) &&
                             (D > 0)>* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef)
  {
    if(accessTarget == access::target::host_buffer)
    {
      this->init_host_accessor(bufferRef);
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
    this->init_device_accessor(bufferRef, commandGroupHandlerRef);
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
    if(accessTarget == access::target::host_buffer)
    {
      this->init_host_accessor(bufferRef);
    }
    else
    {
      // ToDo
      throw unimplemented{"Placeholder accessors are unsupported"};
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
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset = {})
  {
    this->init_device_accessor(bufferRef, commandGroupHandlerRef);
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
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::global_buffer>>
  __host__ __device__
  global_ptr<dataT> get_pointer() const
  {
    return global_ptr<dataT>{_ptr};
  }

  /* Available only when: accessTarget == access::target::constant_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::constant_buffer>>
  __host__ __device__
  constant_ptr<dataT> get_pointer() const
  {
    return constant_ptr<dataT>{_ptr};
  }
private:
  template<class Buffer_type>
  void init_device_accessor(Buffer_type& bufferRef,
                            handler& commandGroupHandlerRef)
  {
    detail::buffer_ptr buff = detail::buffer::get_buffer_impl(bufferRef);
    this->_ptr = reinterpret_cast<pointer_type>(
          detail::accessor::obtain_device_access(buff,
                                                 commandGroupHandlerRef,
                                                 accessmode));

    this->_range = detail::buffer::get_buffer_range(bufferRef);
  }

  template<class Buffer_type>
  void init_host_accessor(Buffer_type& bufferRef)
  {
    detail::buffer_ptr buff = detail::buffer::get_buffer_impl(bufferRef);
    this->_ptr = reinterpret_cast<pointer_type>(
          detail::accessor::obtain_host_access(buff,
                                               accessmode));

    this->_range = detail::buffer::get_buffer_range(bufferRef);
  }

  __host__ __device__
  accessor(){}

  range<dimensions> _range;
  pointer_type _ptr;
};

/// Accessor specialization for local memory
template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::placeholder isPlaceholder>
class accessor<
    dataT,
    dimensions,
    accessmode,
    access::target::local,
    isPlaceholder>
{
  using address = detail::local_memory::address;
public:
  static_assert(isPlaceholder == access::placeholder::false_t,
                "Local accessors cannot be placeholders.");

  using value_type = dataT;
  using reference = dataT &;
  using const_reference = const dataT &;


  /* Available only when: dimensions == 0 */
  template<int D = dimensions,
           typename std::enable_if_t<D == 0>* = nullptr>
  accessor(handler &commandGroupHandlerRef)
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,1)}
  {}

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           typename std::enable_if_t<(D > 0)>* = nullptr>
  accessor(range<dimensions> allocationSize,
           handler &commandGroupHandlerRef)
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,
              allocationSize.size())},
      _num_elements{allocationSize}
  {}


  /* -- common interface members -- */
  __device__
  size_t get_size() const
  {
    return get_count() * sizeof(dataT);
  }

  __device__
  size_t get_count() const
  {
    return _num_elements.size();
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions == 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            D == 0>* = nullptr>
  __device__
  operator dataT &() const
  {
    return *detail::local_memory::get_ptr<dataT>(_addr);
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions > 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            (D > 0)>* = nullptr>
  __device__
  dataT &operator[](id<dimensions> index) const
  {
    *(detail::local_memory::get_ptr<dataT>(_addr) +
        detail::linear_id<dimensions>::get(index, _num_elements));
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions == 1) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            D == 1>* = nullptr>
  __device__
  dataT &operator[](size_t index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) + index);
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 0 */
  //operator atomic<dataT,access::address_space::local_space> () const;


  /* Available only when: accessMode == access::mode::atomic && dimensions > 0 */
  //atomic<dataT, access::address_space::local_space> operator[](
  //      id<dimensions> index) const;

  /* Available only when: accessMode == access::mode::atomic && dimensions == 1 */
  //atomic<dataT, access::address_space::local_space> operator[](size_t index) const;

  /* Available only when: dimensions > 1 */
  template<int D = dimensions,
           std::enable_if_t<(D > 1)>* = nullptr>
  __device__
  accessor<dataT, dimensions-1, accessmode, access::target::local, isPlaceholder>&
  operator[](size_t index) const
  {
    // Create sub accessor for slice at _ptr[index][...]

    using subaccessor_type = accessor<
        dataT,
        dimensions-1,
        accessmode,
        access::target::local,
        isPlaceholder>;

    auto subrange =
        detail::range::omit_first_dimension(this->_num_elements);
    address subaddr = this->_addr
                + index * subrange.size() * sizeof(dataT);

    return subaccessor_type{subaddr, subrange};
  }

  __device__
  local_ptr<dataT> get_pointer() const
  {
    return local_ptr<dataT>{
      detail::local_memory::get_ptr<dataT>(_addr)
    };
  }

private:
  __device__
  accessor(address addr, range<dimensions> r)
    : _addr{addr}, _num_elements{r}
  {}

  const address _addr;
  const range<dimensions> _num_elements;
};

namespace detail {
namespace accessor {
/*

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
*/

} // accessor
} // detail


} // sycl
} // cl

#endif
