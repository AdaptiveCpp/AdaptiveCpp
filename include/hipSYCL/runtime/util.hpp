/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_SCHEDULING_UTIL_HPP
#define HIPSYCL_SCHEDULING_UTIL_HPP

#include <cassert>
#include <array>
#include <type_traits>
#include <cstdint>


namespace hipsycl {
namespace rt {

template<class U, class T>
bool dynamic_is(T* val)
{
  return dynamic_cast<U*>(val) != nullptr;
}

template<class U, class T>
void assert_is(T* val)
{
  assert(dynamic_is<U>(val));
}

template<class U, class T>
U* cast(T* val)
{
  assert_is<U>(val);
  return static_cast<U*>(val);
}


template <int Dim> class static_array {
public:

  static_array() = default;
  static_array(const static_array &other) : _data{other._data} {}


  static_array(std::size_t dim0) {
    for (int i = 0; i < Dim; ++i)
      _data[i] = dim0;
  }

  template<int D = Dim,
           typename = std::enable_if_t<D == 2>>
  static_array(std::size_t dim0, std::size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==3 */
  template<int D = Dim,
           typename = std::enable_if_t<D == 3>>
  static_array(std::size_t dim0, std::size_t dim1, std::size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  friend bool operator==(const static_array<Dim> &a,
                         const static_array<Dim> &b) {
    return a._data == b._data;
  }

  friend bool operator!=(const static_array<Dim> &a,
                         const static_array<Dim> &b) {
    return !(a == b);
  }

  std::size_t &operator[](int dim) { return _data[dim]; }
  std::size_t operator[](int dim) const { return _data[dim]; }
  
  std::size_t get(int dim) const { return _data[dim]; }

#define HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(op)                                  \
  static_array<Dim> &operator op(const static_array<Dim> &b) {                 \
    for (int i = 0; i < Dim; ++i)                                              \
      _data[i] op b._data[i];                                                  \
    return *this;                                                              \
  }

  HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(+=)
  HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(-=)
  HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(*=)
  HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(/=)
  HIPSYCL_RT_SARRAY_MAKE_INPLACE_OP(%=)

#define HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(op)                                  \
  friend static_array<Dim> operator op(const static_array<Dim> &a,             \
                                       const static_array<Dim> &b) {           \
    static_array<Dim> result;                                                  \
    for (int i = 0; i < Dim; ++i)                                              \
      result._data[i] = a._data[i] op b._data[i];                              \
    return result;                                                             \
  }

  HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(+)
  HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(-)
  HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(*)
  HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(/)
  HIPSYCL_RT_SARRAY_MAKE_OOPLACE_OP(%)


  std::size_t size() const {
    std::size_t s = 1;
    for (int i = 0; i < Dim; ++i)
      s *= _data[i];
    return s;
  }
private:
  std::array<std::size_t, Dim> _data;
};


template <int Dim> using id = static_array<Dim>;
template <int Dim> using range = static_array<Dim>;

template<int Dim>
id<3> embed_in_id3(id<Dim> idx) {
  static_assert(Dim >= 1 && Dim <=3, 
      "id dim must be between 1 and 3");

  if constexpr(Dim == 1) {
    return id<3>{0, 0, idx[0]};
  } else if constexpr(Dim == 2) {
    return id<3>{0,idx[0], idx[1]};
  } else if constexpr(Dim == 3) {
    return idx;
  }
}

template<int Dim>
range<3> embed_in_range3(range<Dim> r) {
  static_assert(Dim >= 1 && Dim <=3, 
      "range dim must be between 1 and 3");

  if constexpr(Dim == 1) {
    return range<3>{1, 1, r[0]};
  } else if constexpr(Dim == 2) {
    return range<3>{1, r[0], r[1]};
  } else if constexpr(Dim == 3) {
    return r;
  }
}

template <template <int> class CompatibleArray, int Dim>
static_array<Dim> make_static_array(CompatibleArray<Dim> a) {
  static_array<Dim> d;
  for (int i = 0; i < Dim; ++i) {
    d[i] = a[i];
  }
  return d;
}

template <template <int> class CompatibleId, int Dim>
id<Dim> make_id(CompatibleId<Dim> v) {
  return make_static_array(v);
}

template <template <int> class CompatibleRange, int Dim>
range<Dim> make_range(CompatibleRange<Dim> v) {
  return make_static_array(v);
}

template <template <int> class CompatibleId, int Dim>
id<3> embed_in_id3(CompatibleId<Dim> idx) {
  return embed_in_id3(make_static_array(idx));
}

template <template <int> class CompatibleId, int Dim>
range<3> embed_in_range3(CompatibleId<Dim> idx) {
  return embed_in_range3(make_static_array(idx));
}

template<int Dim>
id<Dim> extract_from_id3(id<3> idx) {
  static_assert(Dim > 0 && Dim <= 3, "Dim must be 1,2 or 3");

  if constexpr (Dim == 1) {
    return id<1>{idx[2]};
  } else if constexpr (Dim == 2) {
    return id<2>{idx[1], idx[2]};
  } else {
    return idx;
  }
  return range<Dim>{};
}

template <int Dim> range<Dim> extract_from_range3(range<3> r) {
  static_assert(Dim > 0 && Dim <= 3, "Dim must be 1,2 or 3");

  if constexpr (Dim == 1) {
    return range<1>{r[2]};
  } else if constexpr (Dim == 2) {
    return range<2>{r[1], r[2]};
  } else {
    return r;
  }
  return range<Dim>{};
}

/* borrowed from LLVM
 * Returns the next power of two (in 64-bits) that is strictly greater than \param a.
 * Returns zero on overflow.
 */
inline std::uint64_t next_power_of_2(std::uint64_t a) {
  a |= (a >> 1);
  a |= (a >> 2);
  a |= (a >> 4);
  a |= (a >> 8);
  a |= (a >> 16);
  a |= (a >> 32);
  return a + 1;
}

/*
 * Returns the power of two which is greater than or equal to the given value.
 * Essentially, it is a ceil operation across the domain of powers of two.
 */
inline std::uint64_t power_of_2_ceil(std::uint64_t a) {
  if (a == 0)
    return 0;
  return next_power_of_2(a - 1);
}

}
}

#endif
