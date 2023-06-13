/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay and contributors
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
#include <boost/test/tools/old/interface.hpp>

BOOST_FIXTURE_TEST_SUITE(profiler_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(queue_no_profiling_exception)
{
  const auto is_invalid = [](const cl::sycl::exception &e) {
    return e.code() ==  cl::sycl::errc::invalid;
  };

  cl::sycl::queue queue;  // no enable_profiling
  constexpr size_t n = 4;

  cl::sycl::buffer<int, 1> buf1{cl::sycl::range<1>(1)};
  auto evt1 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class no_profile_single_task>([=]() {
      acc[0] = 42;
    });
  });
  BOOST_CHECK_EXCEPTION(evt1.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);

  auto evt2 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class no_profile_parallel_for>(buf1.get_range(), [=](cl::sycl::item<1> item) {
      acc[item] = item.get_id()[0];
    });
  });

  cl::sycl::buffer<int, 1> buf2{buf1.get_range()};
  auto evt3 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.copy(buf1.get_access<cl::sycl::access::mode::read>(cgh),
             buf2.get_access<cl::sycl::access::mode::discard_write>(cgh));
  });

  BOOST_CHECK_EXCEPTION(evt3.get_profiling_info<cl::sycl::info::event_profiling::command_end>(),
                        cl::sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt2.get_profiling_info<cl::sycl::info::event_profiling::command_start>(),
                        cl::sycl::exception,
                        is_invalid);

  auto evt4 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.fill(buf1.get_access<cl::sycl::access::mode::discard_write>(cgh), 1);
  });

  int host_array[n];
  auto evt5 = queue.submit([&](cl::sycl::handler &cgh) {
    // copy to pointer
    cgh.copy(buf2.get_access<cl::sycl::access::mode::read>(cgh), host_array);
  });

  BOOST_CHECK_EXCEPTION(evt4.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt5.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);
}

BOOST_AUTO_TEST_CASE(queue_profiling)
{
  cl::sycl::queue queue{cl::sycl::property::queue::enable_profiling{}};
  constexpr size_t n = 4;
  cl::sycl::buffer<int, 1> buf1{cl::sycl::range<1>(n)};

  auto evt1 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class profile_single_task>([=]() {
      acc[0] = 42;
    });
  });

  auto t12 = evt1.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t11 = evt1.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t13 = evt1.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t11 <= t12 && t12 <= t13);
  
  auto evt2 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class profile_parallel_for>(buf1.get_range(), [=](cl::sycl::item<1> item) {
      acc[item] = item.get_id()[0];
    });
  });

  cl::sycl::buffer<int, 1> buf2{buf1.get_range()};
  auto evt3 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.copy(buf1.get_access<cl::sycl::access::mode::read>(cgh),
             buf2.get_access<cl::sycl::access::mode::discard_write>(cgh));
  });

  auto t21 = evt2.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t22 = evt2.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t23 = evt2.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t21 <= t22 && t22 <= t23);

  auto t31 = evt3.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t32 = evt3.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t33 = evt3.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t31 <= t32 && t32 <= t33);
  BOOST_CHECK(t21 <= t31 && t23 <= t32);

  auto evt4 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.fill(buf1.get_access<cl::sycl::access::mode::discard_write>(cgh), 1);
  });

  int host_array[n];
  auto evt5 = queue.submit([&](cl::sycl::handler &cgh) {
    // copy to pointer
    cgh.copy(buf2.get_access<cl::sycl::access::mode::read>(cgh), host_array);
  });

  auto t51 = evt5.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t52 = evt5.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t53 = evt5.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t51 <= t52 && t52 <= t53);

  // re-ordered
  auto t41 = evt4.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t42 = evt4.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t43 = evt4.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t41 <= t42 && t42 <= t43);

  // usm
  auto *src = cl::sycl::malloc_shared<int>(n, queue);
  auto *dest = cl::sycl::malloc_shared<int>(n, queue);
  auto evt6 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.memset(src, 0, sizeof src);
  });

  auto t61 = evt6.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t62 = evt6.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t63 = evt6.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t61 <= t62 && t62 <= t63);

  auto evt7 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.memcpy(dest, src, sizeof src);
  });
  auto t71 = evt7.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t72 = evt7.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t73 = evt7.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  BOOST_CHECK(t71 <= t72 && t72 <= t73);

  auto evt8 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.prefetch(dest, sizeof src);
  });
  auto t81 = evt8.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto t82 = evt8.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto t83 = evt8.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  // run time may be zero if prefetching is a no-op
  BOOST_CHECK(t81 <= t82 && t82 <= t83);

  cl::sycl::free(src, queue);
  cl::sycl::free(dest, queue);
}

// update_host is an explicit operation, but implemented as a requirement and thus not profiled.
// remove the warning and merge this test into queue_profiling if that is ever resolved.
BOOST_AUTO_TEST_CASE(queue_profiling_update_host_supported)
{
  cl::sycl::queue queue{cl::sycl::property::queue::enable_profiling{}};
  int i = 1337;
  cl::sycl::buffer<int, 1> buf{&i, cl::sycl::range<1>(1)};
  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class basic_single_task>([=]() {
      acc[0] = 42;
    });
  });
  auto evt = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.update_host(buf.get_access<cl::sycl::access::mode::read>(cgh));
  });
  BOOST_WARN_NO_THROW(evt.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
}


BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
