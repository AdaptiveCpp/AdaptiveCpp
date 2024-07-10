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

  // fill buffer portions that are not written to with a canary value to make sure copy() does not touch them
  const size_t canary = 0xdeadbeef;

  const auto buf_size = make_test_value<s::range, d>(
      {64}, {64, 96}, {64, 96, 48});
  const auto window_range = make_test_value<s::range, d>(
      {32}, {32, 48}, {32, 48, 24});
  const auto window_offset = make_test_value<s::id, d>(
      {16}, {16, 24}, {16, 24, 12});

  // host_buf_full: item = linear_id * 2
  std::vector<size_t> host_buf_full(buf_size.size());
  for(size_t i = 0; i < buf_size[0]; ++i) {
    const auto dim1_stride = (d >= 2 ? buf_size[1] : 1);
    for(size_t j = 0; j < dim1_stride; ++j) {
      const auto dim2_stride = (d == 3 ? buf_size[2] : 1);
      for(size_t k = 0; k < dim2_stride; ++k) {
        const size_t linear_id = i * dim1_stride * dim2_stride + j * dim2_stride + k;
        host_buf_full[linear_id] = linear_id * 2;
      }
    }
  }

  // host_buf_contiguous: values within the window concatenated into contiguous memory
  // host_buf_window: == host_buf_full within window, canary outside of window
  std::vector<size_t> host_buf_contiguous(buf_size.size(), canary);
  std::vector<size_t> host_buf_window(buf_size.size(), canary);
  const auto full_dim0_stride = buf_size[0];
  const auto contiguous_dim0_stride = window_range[0];
  const auto window_dim0_offset = window_offset[0];
  for (size_t i = window_dim0_offset; i < window_dim0_offset + contiguous_dim0_stride; ++i) {
    const auto full_dim1_stride = (d >= 2 ? buf_size[1] : 1);
    const auto contiguous_dim1_stride = (d >= 2 ? window_range[1] : 1);
    const auto window_dim1_offset = (d >= 2 ? window_offset[1] : 0);
    for (size_t j = window_dim1_offset; j < window_dim1_offset + contiguous_dim1_stride; ++j) {
      const auto full_dim2_stride = (d == 3 ? buf_size[2] : 1);
      const auto contiguous_dim2_stride = (d == 3 ? window_range[2] : 1);
      const auto window_dim2_stride = (d == 3 ? window_offset[2] : 0);
      for (size_t k = window_dim2_stride; k < window_dim2_stride + contiguous_dim2_stride; ++k) {
        const size_t id_in_window = i * full_dim1_stride * full_dim2_stride + j * full_dim2_stride + k;
        const size_t id_in_contiguous = (i - window_offset[0]) * contiguous_dim1_stride * contiguous_dim2_stride
            + (j - window_dim1_offset) * contiguous_dim2_stride
            + (k - window_dim2_stride);
        host_buf_contiguous[id_in_contiguous] = host_buf_full[id_in_window];
        host_buf_window[id_in_window] = host_buf_full[id_in_window];
      }
    }
  }

  cl::sycl::queue queue;

  // copy full buffer without strides
  cl::sycl::buffer<size_t, d> device_buf_full{buf_size};
  queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = device_buf_full.template get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(host_buf_full.data(), acc);
  });

  // copy contiguous buffer into window on the device buffer
  cl::sycl::buffer<size_t, d> device_buf_window{buf_size};
  queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = device_buf_window.template get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.fill(acc, canary);
  });
  queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = device_buf_window.template get_access<cl::sycl::access::mode::discard_write>(cgh,
          window_range, window_offset);
      const auto fake_shared_ptr = std::shared_ptr<size_t>(host_buf_contiguous.data(), [](size_t *){});
      cgh.copy(fake_shared_ptr, acc);
  });

  std::vector<size_t> result_full(buf_size.size(), 0);
  std::vector<size_t> result_contiguous(buf_size.size(), canary);
  std::vector<size_t> result_window(buf_size.size(), canary);

  // copy back full buffer without strides
  queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = device_buf_full.template get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc, result_full.data());
  });
  // copy full buffer from device in strides to obtain contiguous buffer
  queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = device_buf_full.template get_access<cl::sycl::access::mode::read>(cgh, window_range, window_offset);
      cgh.copy(acc, result_contiguous.data());
  });
  /// copy window buffer without strides
  queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = device_buf_window.template get_access<cl::sycl::access::mode::read>(cgh);
      const auto fake_shared_ptr = std::shared_ptr<size_t>(result_window.data(), [](size_t *){});
      cgh.copy(acc, fake_shared_ptr);
  });

  queue.wait();

  BOOST_REQUIRE(result_full == host_buf_full);
  BOOST_REQUIRE(result_contiguous == host_buf_contiguous);
  BOOST_REQUIRE(result_window == host_buf_window);
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

BOOST_AUTO_TEST_SUITE_END()
