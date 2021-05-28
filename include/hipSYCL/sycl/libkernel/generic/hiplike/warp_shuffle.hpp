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

#ifdef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_DETAIL_WARP_SHUFFLE
#define HIPSYCL_DETAIL_WARP_SHUFFLE

namespace hipsycl {
namespace sycl {

namespace detail {

#ifdef HIPSYCL_PLATFORM_HIP
template<typename T, typename Operation>
__device__
T apply_on_data(T x, Operation op) {
  constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

  int words[words_no];
  __builtin_memcpy(words, &x, sizeof(T));

#pragma unroll
  for (int i = 0; i < words_no; i++)
    words[i] = op(words[i]);

  T output;
  __builtin_memcpy(&output, words, sizeof(T));

  return output;
}

// implemented based on warp_shuffle_op in rocPRIM
template<typename T>
__device__
T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](int data) { return __shfl(data, id); });
}
template<typename T>
__device__
T shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_up(data, offset); });
}
template<typename T>
__device__
T shuffle_down_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_down(data, offset); });
}
template<typename T>
__device__
T shuffle_xor_impl(T x, int lane_mask) {
  return apply_on_data(x, [lane_mask](int data) { return __shfl_xor(data, lane_mask); });
}

// dpp sharing instruction abstraction based on rocPRIM
// the dpp_ctrl can be found in the GCN3 ISA manual
template<typename T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
__device__
T warp_move_dpp(T x) {
  return apply_on_data(
      x, [=](int data) { return __builtin_amdgcn_update_dpp(0, data, dpp_ctrl, row_mask, bank_mask, bound_ctrl); });
}
#endif // HIPSYCL_PLATFORM_HIP

#ifdef HIPSYCL_PLATFORM_CUDA
constexpr unsigned int AllMask = 0xFFFFFFFF;

// shuffle_impl implemented based on ShuffleIndex in cub
template<typename T, typename Operation, typename std::enable_if_t<(sizeof(T) == sizeof(unsigned char)), int> = 0>
__device__
T apply_on_data(T x, Operation op) {
  T              data     = x;
  unsigned char *data_ptr = reinterpret_cast<unsigned char *>(&data);
  *data_ptr               = op(*data_ptr);
  return data;
}

template<typename T, typename Operation, typename std::enable_if_t<(sizeof(T) == sizeof(unsigned short)), int> = 0>
__device__
T apply_on_data(T x, Operation op) {
  T               data     = x;
  unsigned short *data_ptr = reinterpret_cast<unsigned short *>(&data);
  *data_ptr                = op(*data_ptr);
  return data;
}

template<typename T, typename Operation, typename std::enable_if_t<(sizeof(T) == sizeof(unsigned int)), int> = 0>
__device__
T apply_on_data(T x, Operation op) {
  T             data     = x;
  unsigned int *data_ptr = reinterpret_cast<unsigned int *>(&data);
  *data_ptr              = op(*data_ptr);
  return data;
}

template<typename T, typename Operation,
         typename std::enable_if_t<(sizeof(T) != sizeof(unsigned char) && sizeof(T) != sizeof(unsigned short) &&
                                    sizeof(T) != sizeof(unsigned int)),
                                   int> = 0>
__device__
T apply_on_data(T x, Operation op) {
  constexpr int words_no = (sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);

  unsigned int words[words_no];
  memcpy(words, &x, sizeof(T));

#pragma unroll
  for (int i = 0; i < words_no; i++)
    words[i] = op(words[i]);

  T output;
  memcpy(&output, words, sizeof(T));

  return output;
}

template<typename T>
__device__
T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](int data) { return __shfl_sync(AllMask, data, id); });
}
template<typename T>
__device__
T shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_up_sync(AllMask, data, offset); });
}
template<typename T>
__device__
T shuffle_down_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_down_sync(AllMask, data, offset); });
}
template<typename T>
__device__
T shuffle_xor_impl(T x, int lane_mask) {
  return apply_on_data(x, [lane_mask](int data) { return __shfl_xor_sync(AllMask, data, lane_mask); });
}

#endif // HIPSYCL_PLATFORM_CUDA

} // namespace detail

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_DETAIL_WARP_SHUFFLE

#endif // SYCL_DEVICE_ONLY
