/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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

#ifndef HIPSYCL_SYCL_HPP
#define HIPSYCL_SYCL_HPP


#define CL_SYCL_LANGUAGE_VERSION 121
#define __SYCL_SINGLE_SOURCE__

#include "extensions.hpp"
#include "backend/backend.hpp"
#include "version.hpp"
#include "types.hpp"
#include "exception.hpp"
#include "device_selector.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "range.hpp"
#include "id.hpp"
#include "accessor.hpp"
#include "buffer.hpp"
#include "nd_item.hpp"
#include "multi_ptr.hpp"
#include "group.hpp"
#include "h_item.hpp"
#include "private_memory.hpp"
#include "vec.hpp"
#include "builtin.hpp"
#include "math.hpp"
#include "common_functions.hpp"
#include "geometric_functions.hpp"
#include "atomic.hpp"
#include "program.hpp"
#include "kernel.hpp"

#endif

