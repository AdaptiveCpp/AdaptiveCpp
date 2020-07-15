/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_UNIT_TESTS_HPP
#define HIPSYCL_UNIT_TESTS_HPP

#include <tuple>

#define BOOST_MPL_CFG_GPU_ENABLED // Required for nvcc
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE hipsycl unit tests
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list_c.hpp>
#include <boost/mpl/list.hpp>

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

struct reset_device_fixture {
  ~reset_device_fixture() {
    ::hipsycl::rt::application::reset();
  }
};

// Helper type to construct unique kernel names for all instantiations of
// a templated test case.
template<typename T, int dimensions, typename extra=T>
struct kernel_name {};

#endif // HIPSYCL_UNIT_TESTS_HPP
