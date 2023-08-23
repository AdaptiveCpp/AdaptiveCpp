/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_PSTL_TEST_SUITE_HPP
#define HIPSYCL_PSTL_TEST_SUITE_HPP


struct enable_unified_shared_memory {
  enable_unified_shared_memory() {
    hipsycl::stdpar::unified_shared_memory::pop_disabled();
  }

  ~enable_unified_shared_memory() {
    hipsycl::stdpar::unified_shared_memory::push_disabled();
  }
};


static thread_local int counter = 0;
struct non_trivial_copy {

  non_trivial_copy(){}

  non_trivial_copy(int val)
  : x{val} {}

  non_trivial_copy(const non_trivial_copy& other){
    x = other.x;
    __hipsycl_if_target_host(++counter;)
  }

  non_trivial_copy& operator=(const non_trivial_copy& other) {
    x = other.x;
    __hipsycl_if_target_host(++counter;)
    return *this;
  }

  friend bool operator==(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x == b.x;
  }

  friend bool operator!=(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x != b.x;
  }

  int x;
};

#endif
