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
#ifndef HIPSYCL_PSTL_STDPAR_DEFS_HPP
#define HIPSYCL_PSTL_STDPAR_DEFS_HPP

#ifdef __clang__
#define HIPSYCL_STDPAR_INLINE __attribute__((always_inline))
#define HIPSYCL_STDPAR_NOINLINE __attribute__((noinline))
#define HIPSYCL_STDPAR_ENTRYPOINT \
    HIPSYCL_STDPAR_NOINLINE __attribute__((annotate("hipsycl_stdpar_entrypoint")))
#else
#define HIPSYCL_STDPAR_INLINE
#define HIPSYCL_STDPAR_ENTRYPOINT
#define HIPSYCL_STDPAR_NOINLINE
#endif

#endif
