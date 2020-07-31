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


#ifndef HIPSYCL_ACCESSOR_HPP
#define HIPSYCL_ACCESSOR_HPP

#include <exception>
#include <type_traits>
#include <cassert>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/deferred_pointer.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/data.hpp"

#include "hipSYCL/sycl/device.hpp"
#include "range.hpp"
#include "access.hpp"
#include "item.hpp"
#include "buffer_allocator.hpp"
#include "backend/backend.hpp"
#include "multi_ptr.hpp"
#include "atomic.hpp"
#include "detail/local_memory_allocator.hpp"
#include "detail/mobile_shared_ptr.hpp"

namespace hipsycl {
namespace sycl {


template <typename T, int dimensions = 1,
          typename AllocatorT = sycl::buffer_allocator<T>>
class buffer;

class handler;

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
class accessor;

namespace detail::handler {

template<class T>
inline
detail::local_memory::address allocate_local_mem(sycl::handler&,
                                                 size_t num_elements);

} // detail::handler

namespace detail::accessor {

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


template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget, access::placeholder isPlaceholder,
          int current_dimension = 1>
class subscript_proxy
{
  HIPSYCL_UNIVERSAL_TARGET
  static constexpr bool can_invoke_access(int current_dim, int dim) {
    return current_dim == dim - 1;
  }
public:
  static_assert(dimensions > 1, "dimension must be > 1");
  
  using accessor_type = sycl::accessor<dataT, dimensions, accessmode,
                                       accessTarget, isPlaceholder>;

  using next_subscript_proxy =
      subscript_proxy<dataT, dimensions, accessmode, accessTarget,
                      isPlaceholder, current_dimension+1>;

  HIPSYCL_UNIVERSAL_TARGET
  subscript_proxy(const accessor_type *original_accessor,
                  sycl::id<dimensions> current_access_id)
      : _original_accessor{original_accessor}, _access_id{current_access_id} {}


  template <int D = dimensions,
            int C = current_dimension,
            std::enable_if_t<!can_invoke_access(C, D)> * = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  next_subscript_proxy operator[](size_t index) const {
    return create_next_proxy(index);
  }


  template <int D = dimensions,
            int C = current_dimension,
            access::mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && 
                            (M == access::mode::read || 
                             M == access::mode::atomic)> * = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto operator[](size_t index) const {
    return invoke_value_access(index);
  }

  template <int D = dimensions,
            int C = current_dimension,
            access::mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && 
                            (M != access::mode::read &&
                             M != access::mode::atomic)> * = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto& operator[](size_t index) const {
    return invoke_ref_access(index);
  }
private:
  HIPSYCL_UNIVERSAL_TARGET
  auto invoke_value_access(size_t index) const {
    // Set the last index
    _access_id[dimensions - 1] = index;
    return (*_original_accessor)[_access_id];
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  auto& invoke_ref_access(size_t index) const {
    // Set the last index
    _access_id[dimensions - 1] = index;
    return (*_original_accessor)[_access_id];
  }

  HIPSYCL_UNIVERSAL_TARGET
  next_subscript_proxy create_next_proxy(size_t next_id) const {
    _access_id[current_dimension] = next_id;
    return next_subscript_proxy{_original_accessor, _access_id};
  }

  const accessor_type *_original_accessor;
  mutable sycl::id<dimensions> _access_id;
};

// These two functions are defined in buffer.hpp
template<class AccessorType, class BufferType>
void bind_to_buffer(AccessorType& acc, BufferType& buff);

template<class AccessorType, class BufferType, int Dim>
void bind_to_buffer(AccessorType& acc, BufferType& buff, 
                  sycl::id<Dim> accessOffset, 
                  sycl::range<Dim> accessRange);

// This function is defined in handler.hpp
template<class AccessorType>
void bind_to_handler(AccessorType& acc, sycl::handler& cgh);

} // detail::accessor

namespace detail {

using mobile_buffer_ptr = mobile_shared_ptr<rt::buffer_data_region>;

/// The accessor base allows us to retrieve the associated buffer
/// for the accessor.
template<class T>
class accessor_base
{
protected:
  friend class sycl::handler;

  accessor_base()
#ifndef SYCL_DEVICE_ONLY
  : _ptr{nullptr}
#endif
  {}

#ifndef SYCL_DEVICE_ONLY

  void set_data_region(std::shared_ptr<rt::buffer_data_region> buff) {
    _buff = buff;
  }

  void bind_to(rt::buffer_memory_requirement* req) {
    assert(req);
    _ptr = req->make_deferred_pointer<T>();
  }

  std::shared_ptr<rt::buffer_data_region> get_data_region() const {
    return _buff.get_shared_ptr();
  }
#endif

  mobile_buffer_ptr _buff;
  glue::deferred_pointer<T> _ptr;
};


} // detail

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget = access::target::global_buffer,
         access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor : public detail::accessor_base<dataT>
{
  template<class AccessorType, class BufferType, int Dim>
  friend void detail::accessor::bind_to_buffer(
    AccessorType& acc, BufferType& buff, 
    sycl::id<Dim> accessOffset, sycl::range<Dim> accessRange);
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
  accessor(buffer<dataT, 1> &bufferRef)
  {
    throw unimplemented{"0-dimensional accessors are not yet implemented"};
    detail::accessor::bind_to_buffer(*this, bufferRef);

    if(T == access::target::host_buffer) {
      init_host_buffer();
    }
  }

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
  accessor(buffer<dataT, 1> &bufferRef, handler &commandGroupHandlerRef)
  {
    throw unimplemented{"0-dimensional accessors are not yet implemented"};
    detail::accessor::bind_to_buffer(*this, bufferRef);
  }

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
    detail::accessor::bind_to_buffer(*this, bufferRef);

    if(T == access::target::host_buffer) {
      init_host_buffer();
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
    detail::accessor::bind_to_buffer(*this, bufferRef);
    detail::accessor::bind_to_handler(*this, commandGroupHandlerRef);
  }

  /// Creates an accessor for a partial range of the buffer, described by an offset
  /// and range.
  ///
  ///
  /// Available only when: (isPlaceholder == access::placeholder::false_t &&
  /// accessTarget == access::target::host_buffer) || (isPlaceholder ==
  /// access::placeholder::true_t && (accessTarget == access::target::global_buffer
  /// || accessTarget == access::target::constant_buffer)) && dimensions > 0 */
  template<access::placeholder P = isPlaceholder,
           access::target T = accessTarget,
           int D = dimensions,
           std::enable_if_t<(P == access::placeholder::false_t &&
                             T == access::target::host_buffer) ||
                            ((P == access::placeholder::true_t &&
                            (T == access::target::global_buffer ||
                             T == access::target::constant_buffer)) &&
                            (D > 0)) >* = nullptr>
  accessor(buffer<dataT, dimensions> &bufferRef,
           range<dimensions> accessRange,
           id<dimensions> accessOffset = {})
  {
    detail::accessor::bind_to_buffer(*this, bufferRef, accessOffset,
                                     accessRange);

    if(T == access::target::host_buffer) {
      init_host_buffer();
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
    detail::accessor::bind_to_buffer(*this, bufferRef, accessOffset, accessRange);
    detail::accessor::bind_to_handler(*this, commandGroupHandlerRef);
  }

  accessor(const accessor& other) = default;
  accessor& operator=(const accessor& other) = default;


  /* -- common interface members -- */
  HIPSYCL_UNIVERSAL_TARGET
  constexpr bool is_placeholder() const
  {
    return isPlaceholder == access::placeholder::true_t;
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t get_size() const
  {
    return get_count() * sizeof(dataT);
  }

  template<int D = dimensions,
           typename = std::enable_if_t<(D > 0)>>
  HIPSYCL_UNIVERSAL_TARGET
  size_t get_count() const
  {
    return _range.size();
  }

  template<int D = dimensions,
           std::enable_if_t<D == 0>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  size_t get_count() const
  { return 1; }

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           std::enable_if_t<(D > 0)>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  range<dimensions> get_range() const
  {
    return _range;
  }

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           typename = std::enable_if_t<(D > 0)>>
  HIPSYCL_UNIVERSAL_TARGET
  id<dimensions> get_offset() const
  {
    return _offset;
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
  HIPSYCL_UNIVERSAL_TARGET
  operator dataT &() const
  {
    return *(this->_ptr.get());
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
  HIPSYCL_UNIVERSAL_TARGET
  dataT &operator[](id<dimensions> index) const
  {
    return (this->_ptr.get())[detail::linear_id<dimensions>::get(index, _buffer_range)];
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
  HIPSYCL_UNIVERSAL_TARGET
  dataT &operator[](size_t index) const
  {
    return (this->_ptr.get())[index];
  }


  /* Available only when: accessMode == access::mode::read && dimensions == 0 */
  template<access::mode M = accessmode,
           int D = dimensions,
           typename = std::enable_if_t<M == access::mode::read && D == 0>>
  HIPSYCL_UNIVERSAL_TARGET
  operator dataT() const
  {
    return *(this->_ptr.get());
  }

  /* Available only when: accessMode == access::mode::read && dimensions > 0 */
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D > 0) && (M == access::mode::read)>>
  HIPSYCL_UNIVERSAL_TARGET
  dataT operator[](id<dimensions> index) const
  { return (this->_ptr.get())[detail::linear_id<dimensions>::get(index, _buffer_range)]; }

  /* Available only when: accessMode == access::mode::read && dimensions == 1 */
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D == 1) && (M == access::mode::read)>>
  HIPSYCL_UNIVERSAL_TARGET
  dataT operator[](size_t index) const
  { return (this->_ptr.get())[index]; }


  /* Available only when: accessMode == access::mode::atomic && dimensions == 0*/
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<M == access::mode::atomic && D == 0>>
  HIPSYCL_UNIVERSAL_TARGET
  operator atomic<dataT, access::address_space::global_space> () const
  {
    return atomic<dataT, access::address_space::global_space>{
        global_ptr<dataT>(this->_ptr.get())};
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions > 0*/
  template <int D = dimensions, access::mode M = accessmode,
            typename = std::enable_if_t<(D > 0) && (M == access::mode::atomic)>>
  HIPSYCL_UNIVERSAL_TARGET atomic<dataT, access::address_space::global_space>
  operator[](id<dimensions> index) const 
  {
    return atomic<dataT, access::address_space::global_space>{global_ptr<dataT>(
        this->_ptr.get() +
        detail::linear_id<dimensions>::get(index, _buffer_range))};
  }

  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D == 1) && (M == access::mode::atomic)>>
  HIPSYCL_UNIVERSAL_TARGET
  atomic<dataT, access::address_space::global_space> operator[](size_t index) const
  {
    return atomic<dataT, access::address_space::global_space>{
        global_ptr<dataT>(this->_ptr.get() + index)};
  }

  /* Available only when: dimensions > 1 */
  template <int D = dimensions, typename = std::enable_if_t<(D > 1)>>
  HIPSYCL_UNIVERSAL_TARGET
  detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                    accessTarget, isPlaceholder>
  operator[](size_t index) const
  {
    
    sycl::id<dimensions> initial_index;
    initial_index[0] = index;

    return detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                             accessTarget, isPlaceholder> {
      this, initial_index
    };
  }

  /* Available only when: accessTarget == access::target::host_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T==access::target::host_buffer>>
  dataT *get_pointer() const
  {
    return const_cast<dataT*>(this->_ptr.get());
  }

  /* Available only when: accessTarget == access::target::global_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::global_buffer>>
  HIPSYCL_UNIVERSAL_TARGET
  global_ptr<dataT> get_pointer() const
  {
    return global_ptr<dataT>{const_cast<dataT*>(this->_ptr.get())};
  }

  /* Available only when: accessTarget == access::target::constant_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::constant_buffer>>
  HIPSYCL_UNIVERSAL_TARGET
  constant_ptr<dataT> get_pointer() const
  {
    return constant_ptr<dataT>{const_cast<dataT*>(this->_ptr.get())};
  }
private:

  void init_host_buffer() {
    // TODO: Maybe unify code with handler::update_host()?
    HIPSYCL_DEBUG_INFO << "accessor [host]: Initializing host access" << std::endl;

    assert(this->_buff.get_shared_ptr());
    auto data = this->_buff.get_shared_ptr();

    if(sizeof(dataT) != data->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_node_ptr node;
    {
      rt::dag_build_guard build{rt::application::dag()};

      auto explicit_requirement = rt::make_operation<rt::buffer_memory_requirement>(
          data, _offset, _range, accessmode, accessTarget);

      this->bind_to(
          rt::cast<rt::buffer_memory_requirement>(explicit_requirement.get()));

      rt::execution_hints enforce_bind_to_host;
      enforce_bind_to_host.add_hint(
          rt::make_execution_hint<rt::hints::bind_to_device>(
              detail::get_host_device()));

      node = build.builder()->add_explicit_mem_requirement(
          std::move(explicit_requirement), rt::requirements_list{},
          enforce_bind_to_host);
      
      HIPSYCL_DEBUG_INFO << "accessor [host]: forcing DAG flush for host access..." << std::endl;
      rt::application::dag().flush_sync();
    }
    if(rt::application::get_runtime().errors().num_errors() == 0){
      HIPSYCL_DEBUG_INFO << "accessor [host]: Waiting for completion host access..." << std::endl;

      assert(node);
      node->wait();
    } else {
      HIPSYCL_DEBUG_ERROR << "accessor [host]: Aborting synchronization, "
                             "runtime error list is non-empty"
                          << std::endl;
      glue::throw_asynchronous_errors([](sycl::exception_list errors) {
        glue::print_async_errors(errors);
        // Additionally throw the first exception to create synchronous
        // error behavior
        if (errors.size() > 0) {
          std::rethrow_exception(errors[0]);
        }
      });
    }
    // TODO Need to lock execution of DAG
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  accessor(){}

  range<dimensions> _buffer_range;
  range<dimensions> _range;
  id<dimensions> _offset;
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
  HIPSYCL_KERNEL_TARGET
  size_t get_size() const
  {
    return get_count() * sizeof(dataT);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_count() const
  {
    return _num_elements.size();
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions == 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            D == 0>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  operator dataT &() const
  {
    return *detail::local_memory::get_ptr<dataT>(_addr);
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions > 0) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            (D > 0)>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  dataT &operator[](id<dimensions> index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) +
        detail::linear_id<dimensions>::get(index, _num_elements));
  }

  /* Available only when: accessMode == access::mode::read_write && dimensions == 1) */
  template<access::mode M = accessmode,
           int D = dimensions,
           std::enable_if_t<M == access::mode::read_write &&
                            D == 1>* = nullptr>
  HIPSYCL_KERNEL_TARGET
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
  HIPSYCL_KERNEL_TARGET
  detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                      access::target::local, isPlaceholder>
  operator[](size_t index) const
  {
    sycl::id<dimensions> initial_index;
    initial_index[0] = index;
    
    return detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                             access::target::local, isPlaceholder> {
      this, initial_index
    };
  }

  HIPSYCL_KERNEL_TARGET
  local_ptr<dataT> get_pointer() const
  {
    return local_ptr<dataT>{
      detail::local_memory::get_ptr<dataT>(_addr)
    };
  }

private:
  HIPSYCL_KERNEL_TARGET
  accessor(address addr, range<dimensions> r)
    : _addr{addr}, _num_elements{r}
  {}

  const address _addr;
  const range<dimensions> _num_elements;
};

template <typename dataT, int dimensions = 1>
using local_accessor = accessor<dataT, dimensions, access::mode::read_write,
  access::target::local>;


} // sycl
} // hipsycl

#endif
