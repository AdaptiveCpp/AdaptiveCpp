/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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

#ifndef HIPSYCL_OUTPUT_STREAM_HPP
#define HIPSYCL_OUTPUT_STREAM_HPP

#include <cstdio>

#include "hipSYCL/sycl/libkernel/backend.hpp"

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
  HIPSYCL_UNIVERSAL_TARGET
  stream(size_t totalBufferSize, size_t workItemBufferSize, handler&)
  : _total_buff_size{totalBufferSize}, _work_item_buff_size{workItemBufferSize}
  {}
  /* -- common interface members -- */
  HIPSYCL_UNIVERSAL_TARGET
  size_t get_size() const { return _total_buff_size; }
  
  HIPSYCL_UNIVERSAL_TARGET
  size_t get_work_item_buffer_size() const { return _work_item_buff_size; }
  
  [[deprecated]]
  HIPSYCL_UNIVERSAL_TARGET
  size_t get_max_statement_size() const
  { return get_work_item_buffer_size(); }
  
private:
  size_t _total_buff_size;
  size_t _work_item_buff_size;
};

#if defined(HIPSYCL_PLATFORM_ROCM) && !defined(HIPSYCL_EXPERIMENTAL_ROCM_PRINTF)

template<class T>
[[deprecated("sycl::stream in hipSYCL requires backend printf() support, "
    "printf is however still experimental in AMD ROCm when compiling with clang. "
    "Define HIPSYCL_EXPERIMENTAL_ROCM_PRINTF to attempt to use it. Otherwise, "
    "uses of the stream class will be transformed into no operations.")]]
HIPSYCL_KERNEL_TARGET
const stream& operator<<(const stream& os, T v) {
  return os;
}

#else

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, stream_manipulator manip) {
  if(manip == endl)
    printf("\n");
  // Other stream_manipulators are not yet supported
  return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, char v){
  printf("%c", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned char v){
  printf("%hhu", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, short v){
  printf("%hd", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned short v){
  printf("%hu", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, int v){
  printf("%d", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned int v){
  printf("%u", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, long v){
  printf("%ld", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned long v){
  printf("%lu", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, long long v){
  printf("%lld", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, unsigned long long v){
  printf("%llu", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, char* v) {
  printf("%s", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, const char* v) {
  printf("%s", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, float v){
  printf("%f", v); return os;
}

HIPSYCL_KERNEL_TARGET
inline const stream& operator<<(const stream& os, double v){
  printf("%f", v); return os;
}

template<class T>
HIPSYCL_KERNEL_TARGET
const stream& operator<<(const stream& os, T* v) {
  printf("%p", v); return os;
}

template<class T>
HIPSYCL_KERNEL_TARGET
const stream& operator<<(const stream& os, const T* v){
  printf("%p", v); return os;
}

template<int Dim>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
const stream& operator<<(const stream& os, item<Dim, with_offset> v){
  if constexpr(with_offset)
    os << "item { id: " << v.get_id() << " range: " << v.get_range() 
       << " offset: " << v.get_offset() << "}";
  else
    os << "item { id: " << v.get_id() << " range: " << v.get_range() << "}";
  return os;
}

template<int Dim>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
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
