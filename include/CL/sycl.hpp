/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef SYCL_HPP
#define SYCL_HPP


#define CL_SYCL_LANGUAGE_VERSION 121
#define __SYCL_SINGLE_SOURCE__

#include "sycl/backend/backend.hpp"
#include "sycl/version.hpp"
#include "sycl/types.hpp"
#include "sycl/exception.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/device.hpp"
#include "sycl/platform.hpp"
#include "sycl/queue.hpp"
#include "sycl/range.hpp"
#include "sycl/id.hpp"
#include "sycl/accessor.hpp"
#include "sycl/buffer.hpp"
#include "sycl/nd_item.hpp"
#include "sycl/multi_ptr.hpp"
#include "sycl/group.hpp"
#include "sycl/h_item.hpp"
#include "sycl/private_memory.hpp"
#include "sycl/vec.hpp"
#include "sycl/builtin.hpp"
#include "sycl/math.hpp"
#include "sycl/atomic.hpp"

#endif

