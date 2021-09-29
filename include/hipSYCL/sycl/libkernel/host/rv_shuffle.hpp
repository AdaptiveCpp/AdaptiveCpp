/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HIPSYCL_HAS_RV

#ifndef HIPSYCL_HOST_DETAIL_RV_SHUFFLE
#define HIPSYCL_HOST_DETAIL_RV_SHUFFLE
#include "rv.hpp"

namespace hipsycl
{
namespace sycl
{

namespace detail
{

template <size_t SizeOfT>
void copy_bits(std::uint8_t* tgtPtr, const std::uint8_t* ptr) {
#pragma unroll
  for(int i = 0; i < SizeOfT; ++i)
    tgtPtr[i] = ptr[i];
}

template <typename T, size_t Words>
void copy_bits(std::array<float, Words> &words, T &&x)
{
  copy_bits<sizeof(T)>(reinterpret_cast<std::uint8_t*>(words.data()),
                       reinterpret_cast<std::uint8_t*>(&x));
}

template <typename T, size_t Words>
void copy_bits(T &tgt, const std::array<float, Words> &words)
{
  copy_bits<sizeof(T)>(reinterpret_cast<std::uint8_t*>(&tgt),
                       reinterpret_cast<const std::uint8_t*>(words.data()));
}

template <typename T, typename Operation>
T apply_on_data(T x, Operation &&op)
{
  constexpr std::size_t words_no = (sizeof(T) + sizeof(float) - 1) / sizeof(float);

  std::array<float, words_no> words;
  copy_bits(words, x);

  for(int i = 0; i < words_no; i++)
    words[i] = std::forward<Operation>(op)(words[i]);

  T output;
  copy_bits(output, words);

  return output;
}

// implemented based on warp_shuffle_op in rocPRIM

// difference between shuffle_impl and extract_impl: id for extract must be uniform value.
template <typename T>
HIPSYCL_FORCE_INLINE T shuffle_impl(T x, int id)
{
  return apply_on_data(x, [id](const float data) {
    float ret = data;
    for(int i = 0; i < rv_num_lanes(); ++i)
    {
      const int srcLane = rv_extract(id, i);
      const float v = rv_extract(data, srcLane);
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}
template <typename T>
HIPSYCL_FORCE_INLINE T extract_impl(T x, int id)
{
  return apply_on_data(x, [id](const float data) { return rv_extract(data, id); });
}
template <typename T>
HIPSYCL_FORCE_INLINE T shuffle_up_impl(T x, int offset)
{
  return apply_on_data(x, [offset](const float data) {
    float ret = data;
    for(int i = 0; i < rv_num_lanes(); ++i)
    {
      const float v = rv_extract(data, (rv_num_lanes() - offset + i) % rv_num_lanes());
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}
template <typename T>
HIPSYCL_FORCE_INLINE T shuffle_down_impl(T x, int offset)
{
  return apply_on_data(x, [offset](const float data) {
    float ret = data;
    for(int i = 0; i < rv_num_lanes(); ++i)
    {
      const float v = rv_extract(data, (offset + i) % rv_num_lanes());
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}
template<typename T>
T shuffle_xor_impl(T x, int lane_mask) {
  return apply_on_data(x, [lane_mask](const float data) {
    float ret = data;
    for(int i = 0; i < rv_num_lanes(); ++i)
    {
      int idx = (lane_mask ^ i) & (rv_num_lanes() - 1);
      const float v = rv_extract(data, idx);
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

// dpp sharing instruction abstraction based on rocPRIM
// the dpp_ctrl can be found in the GCN3 ISA manual
// template<typename T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
// T warp_move_dpp(T x) {
// return apply_on_data(
//   x, [=](int data) { return __builtin_amdgcn_update_dpp(0, data, dpp_ctrl, row_mask, bank_mask, bound_ctrl); });
//}
} // namespace detail

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_HOST_DETAIL_RV_SHUFFLE

#endif // HIPSYCL_HAS_RV
