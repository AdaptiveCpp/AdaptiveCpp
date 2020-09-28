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


#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(explicit_copy_tests, reset_device_fixture)

#ifndef HIPSYCL_TEST_NO_3D_COPIES
using explicit_copy_test_dimensions = test_dimensions;
#else
using explicit_copy_test_dimensions = boost::mpl::list_c<int, 1, 2>;
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_host_ptr, _dimensions,
  explicit_copy_test_dimensions::type) {
  namespace s = cl::sycl;
  // Specify type explicitly to workaround Clang bug #45538
  constexpr int d = _dimensions::value;

  const auto buf_size = make_test_value<s::range, d>(
    { 64 }, { 64, 96 }, { 64, 96, 48 });

  std::vector<size_t> host_buf_source(buf_size.size());
  for(size_t i = 0; i < buf_size[0]; ++i) {
    const auto jb = (d >= 2 ? buf_size[1] : 1);
    for(size_t j = 0; j < jb; ++j) {
      const auto kb = (d == 3 ? buf_size[2] : 1);
      for(size_t k = 0; k < kb; ++k) {
        const size_t linear_id = i * jb * kb + j * kb + k;
        host_buf_source[linear_id] = linear_id * 2;
      }
    }
  }

  cl::sycl::queue queue;
  cl::sycl::buffer<size_t, d> buf{buf_size};

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.copy(host_buf_source.data(), acc);
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<kernel_name<class explicit_buffer_copy_host_ptr_mul3, d>>(
      buf_size, [=](cl::sycl::item<d> item) {
        acc[item] = acc[item] * 3;
      });
  });

  std::shared_ptr<size_t> host_buf_result(new size_t[buf_size.size()],
    std::default_delete<size_t[]>());
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.copy(acc, host_buf_result);
  }).wait();

  for(size_t i = 0; i < buf_size[0]; ++i) {
    const auto jb = (d >= 2 ? buf_size[1] : 1);
    for(size_t j = 0; j < jb; ++j) {
      const auto kb = (d == 3 ? buf_size[2] : 1);
      for(size_t k = 0; k < kb; ++k) {
        const size_t linear_id = i * jb * kb + j * kb + k;
        BOOST_REQUIRE(host_buf_result.get()[linear_id] == linear_id * 6);
      }
    }
  }
}

template<int d, typename callback>
void run_two_accessors_copy_test(const callback& copy_cb) {
  namespace s = cl::sycl;

  const auto src_buf_size = make_test_value<s::range, d>(
    { 64 }, { 64, 96 }, { 64, 96, 48 });
  const auto dst_buf_size = make_test_value<s::range, d>(
    { 32 }, { 32, 16 }, { 32, 16, 24 });
  const auto src_offset = make_test_value<s::id, d>(
    { 24 }, { 24, 70 }, { 24, 70, 12 });
  const auto dst_offset = make_test_value<s::range, d>(
    { 8 }, { 8, 10 }, { 8, 10, 16 });
  const auto copy_range = dst_buf_size - dst_offset;

  s::buffer<s::id<d>, d> src_buf{src_buf_size};
  s::buffer<s::id<d>, d> dst_buf{dst_buf_size};
  
  // Initialize buffers using host accessors
  {
    auto src_acc = src_buf.template get_access<s::access::mode::discard_write>();
    auto dst_acc = dst_buf.template get_access<s::access::mode::discard_write>();

    for(size_t i = 0; i < src_buf_size[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? src_buf_size[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? src_buf_size[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({ i }, { i, j }, { i, j, k });
          src_acc[id] = id;
        }
      }
    }
  }

  // Copy part of larger buffer into smaller buffer
  s::queue queue;
  queue.submit([&](s::handler& cgh) {
    copy_cb(cgh, copy_range, src_buf, src_offset, dst_buf, s::id<d>(dst_offset));
  });

  // Validate results
  {
    auto dst_acc = dst_buf.template get_access<s::access::mode::read>();
    for(size_t i = dst_offset[0]; i < dst_buf_size[0]; ++i) {
      const auto ja = d >= 2 ? dst_offset[1] : 0;
      const auto jb = d >= 2 ? dst_buf_size[1] : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? dst_offset[2] : 0;
        const auto kb = d == 3 ? dst_buf_size[2] : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({ i }, { i, j }, { i, j, k });
          assert_array_equality(id - s::id<d>(dst_offset) + src_offset, dst_acc[id]);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_two_accessors_d2d,
  _dimensions, explicit_copy_test_dimensions::type) {
  constexpr auto d = _dimensions::value;
  namespace s = cl::sycl;
  run_two_accessors_copy_test<d>([](s::handler& cgh, s::range<d> copy_range,
    s::buffer<s::id<d>, d>& src_buf, s::id<d> src_offset,
    s::buffer<s::id<d>, d>& dst_buf, s::id<d> dst_offset) {
      auto src_acc = src_buf.template get_access<s::access::mode::read>(
        cgh, copy_range, src_offset);
      auto dst_acc = dst_buf.template get_access<s::access::mode::discard_write>(
        cgh, copy_range, s::id<d>(dst_offset));
      cgh.copy(src_acc, dst_acc);
    });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_two_accessors_h2d,
  _dimensions, explicit_copy_test_dimensions::type) {
  constexpr auto d = _dimensions::value;
  namespace s = cl::sycl;
  run_two_accessors_copy_test<d>([](s::handler& cgh, s::range<d> copy_range,
    s::buffer<s::id<d>, d>& src_buf, s::id<d> src_offset,
    s::buffer<s::id<d>, d>& dst_buf, s::id<d> dst_offset) {
    auto src_acc = src_buf.template get_access<s::access::mode::read>(
      copy_range, src_offset);
    auto dst_acc = dst_buf.template get_access<s::access::mode::discard_write>(
      cgh, copy_range, s::id<d>(dst_offset));
    cgh.copy(src_acc, dst_acc);
  });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_two_accessors_d2h,
  _dimensions, explicit_copy_test_dimensions::type) {
  constexpr auto d = _dimensions::value;
  namespace s = cl::sycl;
  run_two_accessors_copy_test<d>([](s::handler& cgh, s::range<d> copy_range,
    s::buffer<s::id<d>, d>& src_buf, s::id<d> src_offset,
    s::buffer<s::id<d>, d>& dst_buf, s::id<d> dst_offset) {
    auto src_acc = src_buf.template get_access<s::access::mode::read>(
      cgh, copy_range, src_offset);
    auto dst_acc = dst_buf.template get_access<s::access::mode::discard_write>(
      copy_range, s::id<d>(dst_offset));
    cgh.copy(src_acc, dst_acc);
  });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_two_accessors_h2h,
  _dimensions, explicit_copy_test_dimensions::type) {
  constexpr auto d = _dimensions::value;
  namespace s = cl::sycl;
  run_two_accessors_copy_test<d>([](s::handler& cgh, s::range<d> copy_range,
    s::buffer<s::id<d>, d>& src_buf, s::id<d> src_offset,
    s::buffer<s::id<d>, d>& dst_buf, s::id<d> dst_offset) {
    auto src_acc = src_buf.template get_access<s::access::mode::read>(
      copy_range, src_offset);
    auto dst_acc = dst_buf.template get_access<s::access::mode::discard_write>(
      copy_range, s::id<d>(dst_offset));
    cgh.copy(src_acc, dst_acc);
  });
}

BOOST_AUTO_TEST_SUITE_END()