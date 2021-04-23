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


#ifndef HIPSYCL_PROPERTY_HPP
#define HIPSYCL_PROPERTY_HPP

#include <type_traits>
#include <memory>

#include "access.hpp"
#include "types.hpp"
#include "exception.hpp"

namespace hipsycl {
namespace sycl {

class queue;
class context;
class handler;

template<class T, int Dim, class AllocatorT>
class buffer;

template <typename T, int Dim, access_mode M, target Tgt, access::placeholder P>
class accessor;

template<typename T, int Dim, access_mode M>
class host_accessor;

namespace detail {

class property {};
class queue_property : public property {};
class context_property : public property {};
class buffer_property : public property {};
class accessor_property : public property {};
class cg_property : public property {};
class unknown_property : public property {};

template<class SyclObjectT>
struct associated_property_base {
  using type = unknown_property;
};

template<>
struct associated_property_base<queue> {
  using type = queue_property;
};

template<>
struct associated_property_base<context> {
  using type = context_property;
};

template<class T, int Dim, class AllocatorT>
struct associated_property_base<buffer<T, Dim, AllocatorT>> {
  using type = buffer_property;
};

template<typename T, int Dim, access_mode M, target Tgt, access::placeholder P>
struct associated_property_base<accessor<T, Dim, M, Tgt, P>> {
  using type = accessor_property;
};

template<typename T, int Dim, access_mode M>
struct associated_property_base<host_accessor<T, Dim, M>> {
  using type = accessor_property;
};

template <class SyclObjectT>
using associated_property_base_t =
    typename associated_property_base<SyclObjectT>::type;

} // detail

template <typename PropertyT>
struct is_property : public std::integral_constant<
                         bool, std::is_base_of_v<detail::property, PropertyT>> {
};

template<typename PropertyT>
inline constexpr bool is_property_v = is_property<PropertyT>::value;

template <typename PropertyT, typename SyclObjectT>
struct is_property_of
    : public std::integral_constant<
          bool,
          std::is_base_of_v<detail::associated_property_base_t<SyclObjectT>,
                            PropertyT>> {};

template <typename PropertyT, typename SyclObjectT>
inline constexpr bool is_property_of_v =
    is_property_of<PropertyT, SyclObjectT>::value;

class property_list
{
public:

  template <typename... propertyTN,
    std::enable_if_t<(is_property_v<propertyTN> && ...), bool> = true>
  property_list(propertyTN... props)
  {
    init(props...);
  }

  template <typename propertyT>
  bool has_property() const noexcept
  {
    std::size_t id = typeid(propertyT).hash_code();
    for(std::size_t i = 0; i < _props.size(); ++i) {
      if(_props[i]->type_hash == id) {
        return true;
      }
    }
    
    return false;
  }

  template <typename propertyT>
  propertyT get_property() const
  {
    std::size_t id = typeid(propertyT).hash_code();
    for(std::size_t i = 0; i < _props.size(); ++i) {
      if(_props[i]->type_hash == id) {
        return static_cast<property_wrapper<propertyT> *>(_props[i].get())
            ->property;
      }
    }

    throw invalid_object_error{"Property not found"};
  }
private:

  struct type_erased_property {
    type_erased_property(std::size_t id)
    : type_hash{id} {}

    std::size_t type_hash;
    virtual ~type_erased_property(){}
  };

  template<class PropT>
  struct property_wrapper : public type_erased_property {
    property_wrapper(const PropT &p)
        : property{p}, type_erased_property{typeid(PropT).hash_code()} {}

    PropT property;
  };

  using property_ptr = std::shared_ptr<type_erased_property>;

  template<typename... Props>
  void init(Props... props){
    (add_property(props), ...);
  }

  template<class T>
  void add_property(const T& prop)
  {
    auto ptr = property_ptr{new property_wrapper<T>{prop}};
    _props.push_back(ptr);
  }

  std::vector<property_ptr> _props;
};


namespace detail {

class property_carrying_object
{
public:
  property_carrying_object(const property_list& props)
    : _property_list{props}
  {}

  template <typename propertyT>
  bool has_property() const
  {
    return _property_list.has_property<propertyT>();
  }

  template <typename propertyT>
  propertyT get_property() const
  {
    return _property_list.get_property<propertyT>();
  }

private:
  property_list _property_list;
};
} // detail

} // namespace sycl
} // namespace hipsycl

#endif
