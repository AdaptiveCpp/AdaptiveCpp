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

#ifndef HIPSYCL_DEBUG_HPP
#define HIPSYCL_DEBUG_HPP

#define HIPSYCL_DEBUG_LEVEL_NONE 0
#define HIPSYCL_DEBUG_LEVEL_ERROR 1
#define HIPSYCL_DEBUG_LEVEL_WARNING 2
#define HIPSYCL_DEBUG_LEVEL_INFO 3

#ifndef HIPSYCL_DEBUG_LEVEL
#define HIPSYCL_DEBUG_LEVEL HIPSYCL_DEBUG_LEVEL_NONE
#endif

#include <iostream>



#define HIPSYCL_DEBUG_STREAM(level, prefix) \
if(level > HIPSYCL_DEBUG_LEVEL); else std::cout << prefix

#define HIPSYCL_DEBUG_ERROR \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_ERROR, \
                      "[hipSYCL Error] ")


#define HIPSYCL_DEBUG_WARNING \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_WARNING, \
                      "[hipSYCL Warning] ")


#define HIPSYCL_DEBUG_INFO \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_INFO, \
                      "[hipSYCL Info] ")


#endif
