/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_STATIC_PROPERTY_LIST_HPP
#define HIPSYCL_STATIC_PROPERTY_LIST_HPP

#include <type_traits>
#include "hipSYCL/glue/kernel_names.hpp"
#include "libkernel/range.hpp"

namespace hipsycl {
namespace sycl {


namespace detail {

enum class static_property_type {
  reqd_work_group_size,
  reqd_sub_group_size,
  kernel_name
};

struct static_property {};

template <typename PropertyT>
struct is_static_property
    : public std::integral_constant<
          bool, std::is_base_of_v<static_property, PropertyT>> {};

template <typename PropertyT>
inline constexpr bool is_static_property_v =
    is_static_property<PropertyT>::value;

template<typename... Props>
struct static_property_list {

  static constexpr bool has_properties() {
    return sizeof...(Props) > 0;
  }

  template<static_property_type Type>
  static constexpr auto get_property() {
    if constexpr(has_property<Type>()) {
      return maybe_get<Type,Props...>();
    } else {
      return no_match_t{};
    }
  }

  template<template<typename...> class P>
  static constexpr auto get_property() {
    return get_property<P<>::property_type>();
  }

  template<template<std::size_t...> class P>
  static constexpr auto get_property() {
    return get_property<P<>::property_type>();
  }

  template<template<typename...> class P>
  static constexpr bool has_property() {
    return has_property<P<>::property_type>();
  }

  template<template<std::size_t...> class P>
  static constexpr bool has_property() {
    return has_property<P<>::property_type>();
  }
  
  template<class P>
  static constexpr bool has_property() {
    return has_property<P::property_type>();
  }

  template<static_property_type Type>
  static constexpr bool has_property() {
    return ((Type == Props::property_type) || ...);
  }

private:

  struct no_match_t {};

  template<static_property_type Type>
  static constexpr auto maybe_get() {
    return no_match_t{};
  }

  template<static_property_type Type, class P, typename... Rest>
  static constexpr auto maybe_get() {
    if constexpr(Type == P::property_type)
      return P{};
    else
      return maybe_get<Type, Rest...>();
  }

};

template<class StaticPropertyList, class WrappedType>
struct static_property_wrapper {
  using properties = StaticPropertyList;
  using wrapped_type = WrappedType;
  WrappedType data;
};

template<class WrappedType>
struct static_property_wrapper_traits {
  using wrapped_type = WrappedType;
  using static_property_list_type = static_property_list<>;

  static constexpr auto get_wrapped_data(const WrappedType& t){
    return t;
  }
};

template <typename... Args>
struct static_property_wrapper_traits<static_property_wrapper<Args...>> {
  using wrapped_type = typename static_property_wrapper<Args...>::wrapped_type;
  using static_property_list_type =
      typename static_property_wrapper<Args...>::properties;

  static constexpr auto get_wrapped_data(const static_property_wrapper<Args...>& value){
    return value.data;
  }
};
}

template <std::size_t Size0 = 1024, std::size_t Size1 = 0,
          std::size_t Size2 = 0>
struct reqd_work_group_size : public detail::static_property {
  static constexpr detail::static_property_type property_type =
      detail::static_property_type::reqd_work_group_size;

  template<int Dim>
  static constexpr std::size_t get() {
    if constexpr(Dim == 0)
      return Size0;
    else if constexpr(Dim == 1)
      return Size1;
    else
      return Size2;
  }


  static auto get_range() {
    // TODO This is a bit hacky - it might be cleaner
    // to have the sizes as variadic template pack
    if constexpr(get<2>() != 0) {
      return range<3>{get<0>(), get<1>(), get<2>()};
    } else if constexpr(get<1>() != 0) {
      return range<2>{get<0>(), get<1>()};
    } else {
      return range<1>{get<0>()};
    }
  }
};

template<std::size_t Size = 32>
struct reqd_sub_group_size : public detail::static_property {
  static constexpr detail::static_property_type property_type =
      detail::static_property_type::reqd_sub_group_size;
  static constexpr std::size_t get() {return Size;}
};

// TODO: We can unify the old kernel naming mechanism
// with the static properties mechanism.
// template<class Name>
// struct kernel_name : public detail::static_property {
//  static constexpr detail::static_property_type property_type =
//      detail::static_property_type::kernel_name;
//
//  using name = Name;
//  static constexpr kernel_name<Name> get(){return kernel_name<Name>{};}
//};

template<typename... Props>
using kernel_property_list = detail::static_property_list<Props...>;

template<typename... StaticProperties, class KernelBody>
auto attribute(const KernelBody& b) {
  return detail::static_property_wrapper<
      detail::static_property_list<StaticProperties...>, KernelBody>{b};
}

}
}

#endif