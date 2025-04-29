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
#ifndef HIPSYCL_INFO_HPP
#define HIPSYCL_INFO_HPP

#define HIPSYCL_DEFINE_INFO_DESCRIPTOR(param, ret_type)                        \
  struct param { using return_type = ret_type; };

#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "kernel.hpp"
#include "program.hpp"

#define HIPSYCL_SPECIALIZE_GET_INFO(class_name, specialization)                \
  template<>                                                                   \
  inline typename info::class_name::specialization::return_type                \
  sycl::class_name::get_info<info::class_name::specialization>() const

#endif
