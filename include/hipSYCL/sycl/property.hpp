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

#include "types.hpp"
#include "exception.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

class property {
public:
  virtual ~property() {};
};

using property_ptr = shared_ptr_class<property>;

}

class property_list
{
public:

  template <typename... propertyTN>
  property_list(propertyTN... props)
  {
    init(props...);
  }

  template <typename propertyT>
  bool has_property() const
  {
    for(const auto& property_ptr: _props)
    {
      if(dynamic_cast<propertyT*>(property_ptr.get()) != nullptr)
        return true;
    }
    return false;
  }

  template <typename propertyT>
  propertyT get_property() const
  {
    for(const auto& property_ptr: _props)
    {
      propertyT* prop = dynamic_cast<propertyT*>(property_ptr.get());
      if(prop != nullptr)
        return *prop;
    }
    throw invalid_object_error{"Property not found"};
  }
private:
  void init() {}

  template<class T, typename... Other>
  void init(const T& current, Other... others)
  {
    add_property(current);
    init(others...);
  }

  template<class T>
  void init(const T& current)
  {
    add_property(current);
  }

  template<class T>
  void add_property(const T& prop)
  {
    auto ptr = detail::property_ptr{new T{prop}};
    _props.push_back(ptr);
  }

  vector_class<detail::property_ptr> _props;
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
