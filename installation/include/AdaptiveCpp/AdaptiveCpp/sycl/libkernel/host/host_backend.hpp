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

#ifndef ACPP_LIBKERNEL_HOST_BACKEND_HPP
#define ACPP_LIBKERNEL_HOST_BACKEND_HPP

// Any C++ compiler can do that, so this should always work
#define ACPP_LIBKERNEL_COMPILER_SUPPORTS_HOST 1

// We are in the "device pass" for the host target
// if we are not in an actual device pass OR we
// are using a compiler that has a unified pass for
// host and device.
#if !defined(ACPP_LIBKERNEL_DEVICE_PASS) ||                                 \
    ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
#define ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST 1

// TODO: Do we need those still?
#ifndef ACPP_UNIVERSAL_TARGET
#define ACPP_UNIVERSAL_TARGET
#endif

#ifndef ACPP_KERNEL_TARGET
#define ACPP_KERNEL_TARGET
#endif

#ifndef ACPP_HOST_TARGET
#define ACPP_HOST_TARGET
#endif

#else
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST 0
#endif
#endif
