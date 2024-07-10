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
#ifndef HIPSYCL_DETAIL_WARP_SHUFFLE
#define HIPSYCL_DETAIL_WARP_SHUFFLE

namespace hipsycl {
namespace sycl {

namespace detail::hiplike_builtins::detail {

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
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
T __acpp_shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](int data) { return __shfl(data, id); });
}
template<typename T>
__device__
T __acpp_shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_up(data, offset); });
}
template<typename T>
__device__
T __acpp_shuffle_down_impl(T x, int offset) {
  return apply_on_data(x, [offset](int data) { return __shfl_down(data, offset); });
}
template<typename T>
__device__
T __acpp_shuffle_xor_impl(T x, int lane_mask) {
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
#endif // HIP

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
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
T __acpp_shuffle_impl(T x, int id) {
  // nvc++ fails to correctly determine that the lambda needs to be compiled
  // for device exclusively, so mark as __device__.
  return apply_on_data(x, [id] __device__ (int data) { return __shfl_sync(AllMask, data, id); });
}
template<typename T>
__device__
T __acpp_shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset] __device__ (int data) { return __shfl_up_sync(AllMask, data, offset); });
}
template<typename T>
__device__
T __acpp_shuffle_down_impl(T x, int offset) {
  return apply_on_data(x, [offset] __device__ (int data) { return __shfl_down_sync(AllMask, data, offset); });
}
template<typename T>
__device__
T __acpp_shuffle_xor_impl(T x, int lane_mask) {
  return apply_on_data(x, [lane_mask] __device__ (int data) { return __shfl_xor_sync(AllMask, data, lane_mask); });
}

#endif // CUDA

} // namespace detail

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_DETAIL_WARP_SHUFFLE

