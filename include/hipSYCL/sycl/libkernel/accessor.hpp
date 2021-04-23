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


#ifndef HIPSYCL_ACCESSOR_HPP
#define HIPSYCL_ACCESSOR_HPP

#include <exception>
#include <type_traits>
#include <cassert>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/embedded_pointer.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/data.hpp"

#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/buffer_allocator.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "range.hpp"
#include "item.hpp"
#include "multi_ptr.hpp"
#include "atomic.hpp"
#include "detail/local_memory_allocator.hpp"
#include "detail/mobile_shared_ptr.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

struct read_only_tag_t {};
struct read_write_tag_t {};
struct write_only_tag_t {};
struct read_only_host_task_tag_t {};
struct read_write_host_task_tag_t {};
struct write_only_host_task_tag_t {};

template <typename TagT> constexpr access_mode deduce_access_mode() {
  if constexpr (std::is_same_v<TagT, read_only_tag_t> ||
                std::is_same_v<TagT, read_only_host_task_tag_t>) {
    return access_mode::read;
  } else if constexpr (std::is_same_v<TagT, read_write_tag_t> ||
                       std::is_same_v<TagT, read_write_host_task_tag_t>) {
    return access_mode::read_write;
  } else {
    return access_mode::write;
  }
}

template<typename TagT> constexpr target deduce_access_target() {
  if constexpr (std::is_same_v<TagT, read_only_tag_t> ||
                std::is_same_v<TagT, read_write_tag_t> ||
                std::is_same_v<TagT, write_only_tag_t>) {
    return target::device;
  } else {
    return target::host_task;
  }
}

}

inline constexpr detail::read_only_tag_t read_only;
inline constexpr detail::read_write_tag_t read_write;
inline constexpr detail::write_only_tag_t write_only;
inline constexpr detail::read_only_host_task_tag_t read_only_host_task;
inline constexpr detail::read_write_host_task_tag_t read_write_host_task;
inline constexpr detail::write_only_host_task_tag_t write_only_host_task;

template <typename T, int dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
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

template<class T, access_mode M>
struct accessor_data_type {
  using value = T;
};

template<class T>
struct accessor_data_type<T, access_mode::read> {
  using value = const T;
};

template <typename dataT, int dimensions, access_mode accessmode,
          target accessTarget, access::placeholder isPlaceholder,
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
  using reference = typename accessor_type::reference;

  using next_subscript_proxy =
      subscript_proxy<dataT, dimensions, accessmode, accessTarget,
                      isPlaceholder, current_dimension+1>;

  HIPSYCL_UNIVERSAL_TARGET
  subscript_proxy(const accessor_type *original_accessor,
                  sycl::id<dimensions> current_access_id)
      : _original_accessor{original_accessor}, _access_id{current_access_id} {}


  template <int D = dimensions,
            int C = current_dimension,
            std::enable_if_t<!can_invoke_access(C, D), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  next_subscript_proxy operator[](size_t index) const {
    return create_next_proxy(index);
  }

  template <int D = dimensions,
            int C = current_dimension,
            access_mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && (M != access_mode::atomic), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  reference operator[](size_t index) const {
    return invoke_value_access(index);
  }

  template <int D = dimensions,
            int C = current_dimension,
            access_mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && (M == access_mode::atomic), bool> = true>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_UNIVERSAL_TARGET
  auto operator[](size_t index) const {
    return invoke_atomic_value_access(index);
  }

private:
  HIPSYCL_UNIVERSAL_TARGET
  reference invoke_value_access(size_t index) const {
    // Set the last index
    _access_id[dimensions - 1] = index;
    return (*_original_accessor)[_access_id];
  }

  HIPSYCL_UNIVERSAL_TARGET
  auto invoke_atomic_value_access(size_t index) const {
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

/// The accessor base allows us to retrieve the associated buffer
/// for the accessor.
template<class T>
class accessor_base
{
protected:
  friend class sycl::handler;

  HIPSYCL_HOST_TARGET
  void set_data_region(std::shared_ptr<rt::buffer_data_region> buff) {
#ifndef SYCL_DEVICE_ONLY
    _buff = buff;
#endif
  }

  HIPSYCL_HOST_TARGET
  void bind_to(rt::buffer_memory_requirement *req) {
#ifndef SYCL_DEVICE_ONLY
    assert(req);
    req->bind(_ptr.get_uid());
#endif
  }

  // Only valid until the embedded pointer has been initialized
  HIPSYCL_HOST_TARGET
  glue::unique_id get_uid() const {
    return _ptr.get_uid();
  }

  HIPSYCL_HOST_TARGET
  std::shared_ptr<rt::buffer_data_region> get_data_region() const {
#ifndef SYCL_DEVICE_ONLY
    return _buff.get_shared_ptr();
#else
    // Should never be called on device, but suppress warning
    // about function without return value.
    return nullptr;
#endif
  }

  // Will hold the actual USM pointer after scheduling
  glue::embedded_pointer<T> _ptr;
  mobile_shared_ptr<rt::buffer_data_region> _buff;
};

} // detail

namespace property {

struct no_init : public detail::property {};

} // property

inline constexpr property::no_init no_init;

template <typename dataT, int dimensions = 1,
          access_mode accessmode =
              (std::is_const_v<dataT> ? access_mode::read
                                      : access_mode::read_write),
          target accessTarget = target::device,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor : public detail::accessor_base<std::remove_const_t<dataT>> {

  static_assert(!std::is_const_v<dataT> || accessmode == access_mode::read,
    "const accessors are only allowed for read-only accessors");

  template<class AccessorType, class BufferType, int Dim>
  friend void detail::accessor::bind_to_buffer(
    AccessorType& acc, BufferType& buff, 
    sycl::id<Dim> accessOffset, sycl::range<Dim> accessRange);

  // Need to be friends with other accessors for implicit
  // conversion rules
  template <class Data2, int Dim2, access_mode M2, target Tgt2,
            access::placeholder P2>
  friend class accessor;

  friend class sycl::handler;

public:
  using value_type =
      typename detail::accessor::accessor_data_type<dataT, accessmode>::value;
  using reference = value_type &;
  using const_reference = const dataT &;
  // TODO accessor_ptr
  // TODO iterator, const_interator, reverse_iterator, const_reverse_iterator
  // TODO difference_type
  using size_type = size_t;

  using pointer_type = value_type*;

  accessor() = default;

  // 0D accessors
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<D == 0> * = nullptr>
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           const property_list &prop_list = {}) {

    _is_placeholder = true;
    
    init_properties(prop_list);
    
    detail::accessor::bind_to_buffer(*this, bufferRef);

    if(accessTarget == access::target::host_buffer) {
      init_host_buffer();
    }
  }

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<D == 0> * = nullptr>
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {}) {
    init_properties(prop_list);
    
    detail::accessor::bind_to_buffer(*this, bufferRef);
  }

  // Non 0-dimensional accessors
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           const property_list &prop_list = {}) {
    
    _is_placeholder = true;

    init_properties(prop_list);
    detail::accessor::bind_to_buffer(*this, bufferRef);

    if(accessTarget == access::target::host_buffer) {
      init_host_buffer();
    }
  }

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef, TagT tag,
           const property_list &prop_list = {}) 
  : accessor{bufferRef, prop_list} {}

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {}) {

    init_properties(prop_list);
    detail::accessor::bind_to_buffer(*this, bufferRef);
    detail::accessor::bind_to_handler(*this, commandGroupHandlerRef);
  }

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, TagT tag,
           const property_list &prop_list = {})
      : accessor{bufferRef, commandGroupHandlerRef, prop_list} {}

  /* Ranged accessors */

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, const property_list &propList = {})
  : accessor{bufferRef, accessRange, id<dimensions>{}, propList} {}

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, TagT tag,
           const property_list &propList = {})
  : accessor{bufferRef, accessRange, id<dimensions>{}, tag, propList} {}

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset,
           const property_list &propList = {}) {
    
    _is_placeholder = true;

    init_properties(propList);
    detail::accessor::bind_to_buffer(*this, bufferRef, accessOffset,
                                     accessRange);

    if (accessTarget == access::target::host_buffer) {
      init_host_buffer();
    }
  }

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset, TagT tag,
           const property_list &propList = {})
      : accessor{bufferRef, accessRange, accessOffset, propList} {}

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange,
                 id<dimensions>{}, propList} {}

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           TagT tag, const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, propList} {}

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, const property_list &propList = {}) {

    init_properties(propList);
    detail::accessor::bind_to_buffer(*this, bufferRef, accessOffset,
                                     accessRange);
    detail::accessor::bind_to_handler(*this, commandGroupHandlerRef);

    if (accessTarget == access::target::host_buffer) {
      init_host_buffer();
    }
  }

  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, TagT tag,
           const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, accessOffset,
                 propList} {}

  HIPSYCL_UNIVERSAL_TARGET
  accessor(const accessor& other) = default;

  HIPSYCL_UNIVERSAL_TARGET
  accessor& operator=(const accessor& other) = default;

  // Implicit conversion from read-write accessor to const and non-const
  // read-only accessor
  template <access::placeholder P, access_mode M = accessmode,
            std::enable_if_t<M == access_mode::read> * = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  accessor(const accessor<std::remove_const_t<dataT>, dimensions,
                          access_mode::read_write, accessTarget, P> &other)
      : detail::accessor_base<std::remove_const_t<dataT>>{other},
        _buffer_range{other._buffer_range}, _range{other._range},
        _offset{other._offset}, _is_no_init{other._is_no_init},
        _is_placeholder{other._is_placeholder} {}

  /* -- common interface members -- */
  HIPSYCL_UNIVERSAL_TARGET friend bool operator==(const accessor &lhs,
                                                  const accessor &rhs) {
    bool buffer_same = true;

#ifndef SYCL_DEVICE_ONLY
    buffer_same = lhs._buff.get_shared_ptr() == rhs._buff.get_shared_ptr();
#endif

    return lhs._ptr == rhs._ptr && lhs._buffer_range == rhs._buffer_range &&
           lhs._range == rhs._range && lhs._offset == rhs._offset && buffer_same;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const accessor& lhs, const accessor& rhs)
  {
    return !(lhs == rhs);
  }

  HIPSYCL_UNIVERSAL_TARGET
  bool is_placeholder() const
  {
    return _is_placeholder;
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
  
  template<int D = dimensions,
            std::enable_if_t<(D == 0), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  operator reference() const
  {
    return *(this->_ptr.get());
  }

  template<int D = dimensions,
            access::mode M = accessmode,
            std::enable_if_t<(D > 0) && (M != access::mode::atomic), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  reference operator[](id<dimensions> index) const
  {
    return (this->_ptr.get())[detail::linear_id<dimensions>::get(index, _buffer_range)];
  }

  template<int D = dimensions,
            access::mode M = accessmode,
            std::enable_if_t<(D == 1) && (M != access::mode::atomic), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  reference operator[](size_t index) const
  {
    return (this->_ptr.get())[index];
  }


  /* Available only when: accessMode == access::mode::atomic && dimensions == 0*/
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<M == access::mode::atomic && D == 0>>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_UNIVERSAL_TARGET
  operator atomic<dataT, access::address_space::global_space> () const
  {
    return atomic<dataT, access::address_space::global_space>{
        global_ptr<dataT>(this->_ptr.get())};
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions > 0*/
  template <int D = dimensions, access::mode M = accessmode,
            typename = std::enable_if_t<(D > 0) && (M == access::mode::atomic)>>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_UNIVERSAL_TARGET
  atomic<dataT, access::address_space::global_space> operator[](id<dimensions> index) const 
  {
    return atomic<dataT, access::address_space::global_space>{global_ptr<dataT>(
        this->_ptr.get() +
        detail::linear_id<dimensions>::get(index, _buffer_range))};
  }

  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<(D == 1) && (M == access::mode::atomic)>>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_UNIVERSAL_TARGET
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

  access_mode get_effective_access_mode() const {
    access_mode mode = accessmode;

    if(mode == access_mode::atomic){
      mode = access_mode::read_write;
    }

    if(_is_no_init) {
      if(mode == access_mode::write) {
        mode = access_mode::discard_write;
      } else if(mode == access_mode::read_write) {
        mode = access_mode::discard_read_write;
      }
    }

    return mode;
  }

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

      auto explicit_requirement =
          rt::make_operation<rt::buffer_memory_requirement>(
              data, rt::make_id(_offset), rt::make_range(_range),
              get_effective_access_mode(), accessTarget);

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
      HIPSYCL_DEBUG_INFO << "accessor [host]: Waiting for completion of host access..." << std::endl;

      assert(node);
      node->wait();

      rt::buffer_memory_requirement *req =
          static_cast<rt::buffer_memory_requirement *>(node->get_operation());
      assert(req->has_device_ptr());
      void* host_ptr = req->get_device_ptr();
      assert(host_ptr);
      
      // For host accessors, we need to manually trigger the initialization
      // of the embedded pointer
      this->_ptr.explicit_init(host_ptr);
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

  void init_properties() {
    if(accessmode == access_mode::discard_write ||
       accessmode == access_mode::discard_read_write) {
      _is_no_init = true;
    }
  }

  void init_properties(const property_list& prop_list) {
    init_properties();
    if(prop_list.has_property<property::no_init>()) {
      _is_no_init = true;
    }
  }

  range<dimensions> _buffer_range;
  range<dimensions> _range;
  id<dimensions> _offset;
  bool _is_no_init = false;
  bool _is_placeholder = false;
};

// Accessor deduction guides

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, TagT tag,
          const property_list &prop_list = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::true_t>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         TagT tag, const property_list &prop_list = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::false_t>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::true_t>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::true_t>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::false_t>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, id<Dim> accessOffset, TagT tag,
         const property_list &propList = {})
    -> accessor<T, Dim, detail::deduce_access_mode<TagT>(),
                detail::deduce_access_target<TagT>(),
                access::placeholder::false_t>;

//host_accessor implementation

template <typename dataT, int dimensions = 1,
          access_mode accessMode =
              (std::is_const_v<dataT> ? access_mode::read
                                      : access_mode::read_write)>
class host_accessor {
  using accessor_type =
      accessor<dataT, dimensions, accessMode, target::host_buffer,
               access::placeholder::false_t>;

  template<typename DataT2, int Dim2, access_mode M2>
  friend class host_accessor;

  template<class TagT>
  void validate_host_accessor_tag(TagT tag) {
    static_assert(std::is_same_v<TagT, detail::read_only_tag_t> ||
                  std::is_same_v<TagT, detail::write_only_tag_t> ||
                  std::is_same_v<TagT, detail::read_write_tag_t>,
                  "Invalid tag for host_accessor");
  }
public:
  using value_type = typename accessor_type::value_type;
  using reference = typename accessor_type::reference;
  using const_reference = typename accessor_type::const_reference;

  // using iterator = __unspecified_iterator__<value_type>;
  // using const_iterator = __unspecified_iterator__<const value_type>;
  // using reverse_iterator = std::reverse_iterator<iterator>;
  // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // using difference_type = typename
  // std::iterator_traits<iterator>::difference_type;
  using size_type = typename accessor_type::size_type;

  host_accessor() = default;

  /* Available only when: (dimensions == 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<D == 0, bool> = true>
  host_accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
                const property_list &propList = {})
      : _impl{bufferRef, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                const property_list &propList = {})
      : _impl{bufferRef, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef, TagT tag,
                const property_list &propList = {})
      : _impl{bufferRef, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange, TagT tag,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange, id<dimensions> accessOffset,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, accessOffset, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange, id<dimensions> accessOffset,
                TagT tag, const property_list &propList = {})
      : _impl{bufferRef, accessRange, accessOffset, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  // Conversion read-write -> read-only accessor
  template <access_mode M = accessMode,
            std::enable_if_t<M == access_mode::read, bool> = true>
  host_accessor(const host_accessor<std::remove_const_t<dataT>, dimensions,
                                    access_mode::read_write> &other)
      : _impl{other._impl} {}

  /* -- common interface members -- */

  //void swap(host_accessor &other);

  //size_type byte_size() const noexcept;

  //size_type size() const noexcept;

  //size_type max_size() const noexcept;

  //bool empty() const noexcept;

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  range<dimensions> get_range() const {
    return _impl.get_range();
  }

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  id<dimensions> get_offset() const {
    return _impl.get_offset();
  }

  /* Available only when: (dimensions == 0) */
  template<int D = dimensions,
            std::enable_if_t<(D == 0), bool> = true>
  operator reference() const {
    return *_impl.get_pointer();
  }

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  reference operator[](id<dimensions> index) const {
    return _impl[index];
  }

  /* Available only when: (dimensions > 1) */
  template<int D = dimensions,
            std::enable_if_t<(D > 1), bool> = true>
  auto operator[](size_t index) const {
    return _impl[index];
  }

  /* Available only when: (dimensions == 1) */
  template<int D = dimensions,
            std::enable_if_t<(D == 1), bool> = true>
  reference operator[](size_t index) const {
    return _impl[index];
  }

  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return _impl.get_pointer();
  }

  // iterator begin() const noexcept;
  // iterator end() const noexcept;
  // const_iterator cbegin() const noexcept;
  // const_iterator cend() const noexcept;
  // reverse_iterator rbegin() const noexcept;
  // reverse_iterator rend() const noexcept;
  // const_reverse_iterator crbegin() const noexcept;
  // const_reverse_iterator crend() const noexcept;

private:
  accessor_type _impl;
};


// host_accessor deduction guides

template <typename T, int Dim, typename AllocatorT, typename TagT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, TagT tag,
          const property_list &prop_list = {})
    -> host_accessor<T, Dim, detail::deduce_access_mode<TagT>()>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
              TagT tag, const property_list &propList = {})
    -> host_accessor<T, Dim, detail::deduce_access_mode<TagT>()>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, TagT tag, const property_list &propList = {})
    -> host_accessor<T, Dim, detail::deduce_access_mode<TagT>()>;

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

  using value_type =
      typename detail::accessor::accessor_data_type<dataT, accessmode>::value;
  using reference = value_type &;
  using const_reference = const dataT &;
  // TODO iterator, const_interator, reverse_iterator, const_reverse_iterator
  // TODO difference_type
  using size_type = size_t;


  accessor() = default;

  /* Available only when: dimensions == 0 */
  template<int D = dimensions,
           typename std::enable_if_t<D == 0>* = nullptr>
  accessor(handler &commandGroupHandlerRef, const property_list& p = {})
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,1)}
  {}

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           typename std::enable_if_t<(D > 0)>* = nullptr>
  accessor(range<dimensions> allocationSize,
           handler &commandGroupHandlerRef, const property_list& p = {})
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,
              allocationSize.size())},
      _num_elements{allocationSize}
  {}


  /* -- common interface members -- */

  friend bool operator==(const accessor& lhs, const accessor& rhs)
  {
    return lhs._addr == rhs._addr && lhs._num_elements == rhs._num_elements;
  }

  friend bool operator!=(const accessor& lhs, const accessor& rhs)
  {
    return !(lhs == rhs);
  }

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

  
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 0) && (M != access_mode::atomic), bool> = false>
  HIPSYCL_KERNEL_TARGET
  operator reference() const
  {
    return *detail::local_memory::get_ptr<dataT>(_addr);
  }

  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D > 0) && (M != access_mode::atomic), bool> = false>
  HIPSYCL_KERNEL_TARGET
  reference operator[](id<dimensions> index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) +
        detail::linear_id<dimensions>::get(index, _num_elements));
  }

  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 1) && (M != access_mode::atomic), bool> = false>
  HIPSYCL_KERNEL_TARGET
  reference operator[](size_t index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) + index);
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 0 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 0) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_KERNEL_TARGET
  operator atomic<dataT,access::address_space::local_space>() const
  {
    return atomic<dataT, access::address_space::local_space>{
            local_ptr<dataT>{detail::local_memory::get_ptr<dataT>(_addr)}};
  }


  /* Available only when: accessMode == access::mode::atomic && dimensions > 0 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D > 0) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_KERNEL_TARGET
  atomic<dataT, access::address_space::local_space> operator[](
       id<dimensions> index) const
  {
    return atomic<dataT, access::address_space::local_space>{local_ptr<dataT>{
            detail::local_memory::get_ptr<dataT>(_addr) +
            detail::linear_id<dimensions>::get(index, _num_elements)}};
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 1 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 1) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] HIPSYCL_KERNEL_TARGET
  atomic<dataT, access::address_space::local_space> operator[](size_t index) const
  {
    return atomic<dataT, access::address_space::local_space>{local_ptr<dataT>{
            detail::local_memory::get_ptr<dataT>(_addr) + index}};
  }

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
