/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_OUTPUT_STREAM_HPP
#define HIPSYCL_OUTPUT_STREAM_HPP

#include <cstdio>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/sycl/libkernel/sscp/builtins/print.hpp"
#endif

#include "id.hpp"
#include "range.hpp"
#include "item.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "group.hpp"
#include "h_item.hpp"
#include "multi_ptr.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename... Args>
void print(const char* s, Args... args) {
  __acpp_backend_switch(
    printf(s, args...),
    if constexpr(sizeof...(args) == 0) {
      __acpp_sscp_print(s);
    } else {
      __acpp_sscp_print("Type not yet supported for printing with generic target\n");
    },
    printf(s, args...),
    printf(s, args...));
}

}

enum class stream_manipulator {
  flush,
  dec,
  hex,
  oct,
  noshowbase,
  showbase,
  noshowpos,
  showpos,
  endl,
  fixed,
  scientific,
  hexfloat,
  defaultfloat
};

const stream_manipulator flush = stream_manipulator::flush;
const stream_manipulator dec = stream_manipulator::dec;
const stream_manipulator hex = stream_manipulator::hex;
const stream_manipulator oct = stream_manipulator::oct;
const stream_manipulator noshowbase = stream_manipulator::noshowbase;
const stream_manipulator showbase = stream_manipulator::showbase;
const stream_manipulator noshowpos = stream_manipulator::noshowpos;
const stream_manipulator showpos = stream_manipulator::showpos;
const stream_manipulator endl = stream_manipulator::endl;
const stream_manipulator fixed = stream_manipulator::fixed;
const stream_manipulator scientific = stream_manipulator::scientific;
const stream_manipulator hexfloat = stream_manipulator::hexfloat;
const stream_manipulator defaultfloat = stream_manipulator::defaultfloat;
//__precision_manipulator__ setprecision(int precision);
//__width_manipulator__ setw(int width);

class stream {
public:
  ACPP_UNIVERSAL_TARGET
  stream(size_t totalBufferSize, size_t workItemBufferSize, handler&)
  : _total_buff_size{totalBufferSize}, _work_item_buff_size{workItemBufferSize}
  {}
  /* -- common interface members -- */
  ACPP_UNIVERSAL_TARGET
  size_t get_size() const { return _total_buff_size; }
  
  ACPP_UNIVERSAL_TARGET
  size_t get_work_item_buffer_size() const { return _work_item_buff_size; }
  
  [[deprecated]]
  ACPP_UNIVERSAL_TARGET
  size_t get_max_statement_size() const
  { return get_work_item_buffer_size(); }
  
private:
  size_t _total_buff_size;
  size_t _work_item_buff_size;
};

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP &&                                    \
    !defined(HIPSYCL_EXPERIMENTAL_ROCM_PRINTF)

template<class T>
[[deprecated("sycl::stream in hipSYCL requires backend printf() support, "
    "printf is however still experimental in AMD ROCm when compiling with clang. "
    "Define HIPSYCL_EXPERIMENTAL_ROCM_PRINTF to attempt to use it. Otherwise, "
    "uses of the stream class will be transformed into no operations.")]]
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, T v) {
  return os;
}

#else

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, stream_manipulator manip) {
  if(manip == endl)
    detail::print("\n");
  // Other stream_manipulators are not yet supported
  return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, char v){
  detail::print("%c", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned char v){
  detail::print("%hhu", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, short v){
  detail::print("%hd", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned short v){
  detail::print("%hu", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, int v){
  detail::print("%d", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned int v){
  detail::print("%u", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, long v){
  detail::print("%ld", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned long v){
  detail::print("%lu", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, long long v){
  detail::print("%lld", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned long long v){
  detail::print("%llu", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, char* v) {
  detail::print(v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, const char* v) {
  detail::print(v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, float v){
  detail::print("%f", v); return os;
}

ACPP_KERNEL_TARGET
inline const stream& operator<<(const stream& os, double v){
  detail::print("%f", v); return os;
}

template<class T>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, T* v) {
  detail::print("%p", v); return os;
}

template<class T>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, const T* v){
  detail::print("%p", v); return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, id<Dim> v){
  if constexpr(Dim >= 1)
    os << v[0];
  if constexpr(Dim >= 2)
    os << ", " << v[1];
  if constexpr(Dim >= 3)
    os << ", " << v[2];
  return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, range<Dim> v){
  if constexpr(Dim >= 1)
    os << v[0];
  if constexpr(Dim >= 2)
    os << ", " << v[1];
  if constexpr(Dim >= 3)
    os << ", " << v[2];
  return os;
}

template<int Dim, bool with_offset>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, item<Dim, with_offset> v){
  if constexpr(with_offset)
    os << "item { id: " << v.get_id() << " range: " << v.get_range() 
       << " offset: " << v.get_offset() << "}";
  else
    os << "item { id: " << v.get_id() << " range: " << v.get_range() << "}";
  return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, nd_item<Dim> v){
  os << "nd_item {" 
     << " local_id: "     << v.get_local_id()
     << " local_range: "  << v.get_local_range()
     << " group_id: "     << v.get_group().get_id()
     << " group_range: "  << v.get_group_range()
     << " global_id: "    << v.get_global_id()
     << " global_range: " << v.get_global_range()
     << " offset: "       << v.get_offset()
     << "}";
  return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, nd_range<Dim> v){
    os << "nd_range {"
     << " local_range: "  << v.get_local_range()
     << " group_range: "  << v.get_group()
     << " global_range: " << v.get_global_range()
     << " offset: "       << v.get_offset()
     << "}";
  return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, group<Dim> v){
  os << "group {" 
     << " group_id: "     << v.get_id()
     << " group_range: "  << v.get_group_range()
     << " local_range: "  << v.get_local_range()
     << " global_range: " << v.get_global_range()
     << "}";
  return os;
}

template<int Dim>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, h_item<Dim> v){
  os << "h_item {" 
     << " logical_local_id: "  << v.get_logical_local()
     << " logical_local_range: "  << v.get_logical_local_range()
     << " physical_local_id: " << v.get_physical_local()
     << " physical_local_range: "  << v.get_physical_local_range()
     << " global_id: "         << v.get_global_id()
     << " global_range: "      << v.get_global_range()
     << "}";
  return os;
}


template <typename ElementType, access::address_space Space>
ACPP_KERNEL_TARGET
const stream& operator<<(const stream& os, multi_ptr<ElementType, Space> v){
  
  if constexpr(Space == access::address_space::global_space)
    os << "global_memory@";
  else if constexpr(Space == access::address_space::local_space)
    os << "local_memory@";
  else if constexpr(Space == access::address_space::constant_space)
    os << "constant_memory@";
  else if constexpr(Space == access::address_space::private_space)
    os << "private_memory@";
  
  os << v.get();
  
  return os;
}

#endif

}
}

#endif
