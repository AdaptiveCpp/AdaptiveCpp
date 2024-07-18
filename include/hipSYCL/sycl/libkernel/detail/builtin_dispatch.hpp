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

#ifndef ACPP_LIBKERNEL_BUILTIN_DISPATCH_HPP
#define ACPP_LIBKERNEL_BUILTIN_DISPATCH_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#define HIPSYCL_DISPATCH_BUILTIN(name, ...)                                    \
  __acpp_backend_switch(                                                    \
      host_builtins::name(__VA_ARGS__), sscp_builtins::name(__VA_ARGS__),      \
      hiplike_builtins::name(__VA_ARGS__),                                     \
      hiplike_builtins::name(__VA_ARGS__))

#define HIPSYCL_RETURN_DISPATCH_BUILTIN(name, ...)                             \
  __acpp_backend_switch(return host_builtins::name(__VA_ARGS__),            \
                                  return sscp_builtins::name(__VA_ARGS__),     \
                                  return hiplike_builtins::name(__VA_ARGS__),  \
                                  return hiplike_builtins::name(__VA_ARGS__))

#endif
