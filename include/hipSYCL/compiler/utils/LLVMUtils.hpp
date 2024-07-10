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
#ifndef HIPSYCL_LLVMUTILS_HPP
#define HIPSYCL_LLVMUTILS_HPP


#if LLVM_VERSION_MAJOR < 16
#define IS_OPAQUE(pointer) (pointer->isOpaquePointerTy())
#define HAS_TYPED_PTR 1
#else
#define IS_OPAQUE(pointer) constexpr(true || pointer) /* Use `pointer` to silence warnings */
#define HAS_TYPED_PTR 0
#endif

#endif // HIPSYCL_LLVMUTILS_HPP
