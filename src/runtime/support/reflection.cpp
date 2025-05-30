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

#include "hipSYCL/runtime/support/reflection.hpp"


namespace hipsycl::rt::support {
  symbol_information& symbol_information::get() {
    static symbol_information si;
    return si;
  }
}


// Compiler will generate calls to this function to register functions
extern "C" void __acpp_reflection_associate_function_pointer(const void *func_ptr,
                                           const char *func_name) {
  hipsycl::rt::support::symbol_information::get()
      .register_function_symbol(func_ptr, func_name);
}
