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
#include <boost/test/tools/old/interface.hpp>
#include <chrono>

BOOST_FIXTURE_TEST_SUITE(profiler_tests, reset_device_fixture)

bool default_device_supports_profiling() {
  sycl::queue q;
  return q.get_device().get_info<sycl::info::device::queue_profiling>();
}

BOOST_AUTO_TEST_CASE(queue_no_profiling_exception)
{
  const auto is_invalid = [](const sycl::exception &e) {
    return e.code() ==  sycl::errc::invalid;
  };

  sycl::queue queue;  // no enable_profiling

  sycl::buffer<int, 1> buf1{sycl::range<1>(1)};
  auto evt1 = queue.submit([&](sycl::handler &cgh) {
    auto acc = buf1.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class no_profile_single_task>([=]() {
      acc[0] = 42;
    });
  });
  BOOST_CHECK_EXCEPTION(evt1.get_profiling_info<sycl::info::event_profiling::command_submit>(),
                        sycl::exception,
                        is_invalid);

  auto evt2 = queue.submit([&](sycl::handler &cgh) {
    auto acc = buf1.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class no_profile_parallel_for>(buf1.get_range(), [=](sycl::item<1> item) {
      acc[item] = item.get_id()[0];
    });
  });

  sycl::buffer<int, 1> buf2{buf1.get_range()};
  auto evt3 = queue.submit([&](sycl::handler &cgh) {
    cgh.copy(buf1.get_access<sycl::access::mode::read>(cgh),
             buf2.get_access<sycl::access::mode::discard_write>(cgh));
  });

  BOOST_CHECK_EXCEPTION(evt3.get_profiling_info<sycl::info::event_profiling::command_end>(),
                        sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt2.get_profiling_info<sycl::info::event_profiling::command_start>(),
                        sycl::exception,
                        is_invalid);

  auto evt4 = queue.submit([&](sycl::handler &cgh) {
    cgh.fill(buf1.get_access<sycl::access::mode::discard_write>(cgh), 1);
  });

  int host_array;
  auto evt5 = queue.submit([&](sycl::handler &cgh) {
    // copy to pointer
    cgh.copy(buf2.get_access<sycl::access::mode::read>(cgh), &host_array);
  });
  
  BOOST_CHECK_EXCEPTION(evt4.get_profiling_info<sycl::info::event_profiling::command_submit>(),
                        sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt5.get_profiling_info<sycl::info::event_profiling::command_submit>(),
                        sycl::exception,
                        is_invalid);
  queue.wait();
}

BOOST_AUTO_TEST_CASE(queue_profiling)
{
  if(default_device_supports_profiling()) {
    sycl::queue queue{sycl::property::queue::enable_profiling{}};
    constexpr size_t n = 4;
    sycl::buffer<int, 1> buf1{sycl::range<1>(n)};

    auto evt1 = queue.submit([&](sycl::handler &cgh) {
      auto acc = buf1.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.single_task<class profile_single_task>([=]() { acc[0] = 42; });
    });

    auto t12 = evt1.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t11 = evt1.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t13 =
        evt1.get_profiling_info<sycl::info::event_profiling::command_end>();
    // We cannot test that submit time is <= command start time, since
    // in some backends (e.g. CUDA) time is only measured as elapsed time
    // between two points in low-precision float. Submission time on the other
    // hand is always exact. So there might be rounding errors causing
    // t12 > t11.
    // The same thing could in principle happen when comparing submission time
    // with command end time, but hopefully this is less likely.
    //
    // Additionally measurements have shown that on discrete GPUs the submission
    // timestamp is unreliable due to a drift between host and the device clock
    // therefore we disable these tests.
    auto check_submit_cmdbegin_cmdend = [&]( uint64_t submission,uint64_t buffer_begin, uint64_t buffer_end){
      auto backend = queue.get_device().get_backend();
      if (backend == sycl::backend::cuda || backend == sycl::backend::hip) {
        return buffer_begin <= buffer_end;
      }else {
        return submission <= buffer_end && buffer_begin <= buffer_end;
      }
    };

    BOOST_CHECK(check_submit_cmdbegin_cmdend(t11, t12, t13));

    auto evt2 = queue.submit([&](sycl::handler &cgh) {
      auto acc = buf1.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for<class profile_parallel_for>(
          buf1.get_range(),
          [=](sycl::item<1> item) { acc[item] = item.get_id()[0]; });
    });

    sycl::buffer<int, 1> buf2{buf1.get_range()};
    auto evt3 = queue.submit([&](sycl::handler &cgh) {
      cgh.copy(buf1.get_access<sycl::access::mode::read>(cgh),
               buf2.get_access<sycl::access::mode::discard_write>(cgh));
    });

    auto t21 = evt2.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t22 = evt2.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t23 =
        evt2.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t21, t22, t23));

    auto t31 = evt3.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t32 = evt3.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t33 =
        evt3.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t31, t32, t33));
    BOOST_CHECK(t21 <= t31 && t23 <= t32);

    auto evt4 = queue.submit([&](sycl::handler &cgh) {
      cgh.fill(buf1.get_access<sycl::access::mode::discard_write>(cgh), 1);
    });

    int host_array[n];
    auto evt5 = queue.submit([&](sycl::handler &cgh) {
      // copy to pointer
      cgh.copy(buf2.get_access<sycl::access::mode::read>(cgh), host_array);
    });

    auto t51 = evt5.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t52 = evt5.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t53 =
        evt5.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t51, t52, t53));

    // re-ordered
    auto t41 = evt4.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t42 = evt4.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t43 =
        evt4.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t41, t42, t43));

    // usm
    const bool use_shared_alloc =
        queue.get_device().has(sycl::aspect::usm_shared_allocations);
    auto *src = use_shared_alloc ? sycl::malloc_shared<int>(n, queue)
                                 : sycl::malloc_device<int>(n, queue);
    auto *dest = use_shared_alloc ? sycl::malloc_shared<int>(n, queue)
                                  : sycl::malloc_device<int>(n, queue);
    auto evt6 = queue.submit(
        [&](sycl::handler &cgh) { cgh.memset(src, 0, sizeof src); });

    auto t61 = evt6.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t62 = evt6.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t63 =
        evt6.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t61, t62, t63));

    auto evt7 = queue.submit(
        [&](sycl::handler &cgh) { cgh.memcpy(dest, src, sizeof src); });
    auto t71 = evt7.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t72 = evt7.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t73 =
        evt7.get_profiling_info<sycl::info::event_profiling::command_end>();
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t71, t72, t73));

    auto evt8 = queue.submit(
        [&](sycl::handler &cgh) { cgh.prefetch(dest, sizeof src); });
    auto t81 = evt8.get_profiling_info<
        sycl::info::event_profiling::command_submit>();
    auto t82 = evt8.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto t83 =
        evt8.get_profiling_info<sycl::info::event_profiling::command_end>();
    // run time may be zero if prefetching is a no-op
    BOOST_CHECK(check_submit_cmdbegin_cmdend(t81, t82, t83));

    sycl::free(src, queue);
    sycl::free(dest, queue);
  }
}

// update_host is an explicit operation, but implemented as a requirement and thus not profiled.
// remove the warning and merge this test into queue_profiling if that is ever resolved.
BOOST_AUTO_TEST_CASE(queue_profiling_update_host_supported)
{
  if(default_device_supports_profiling()) {
    sycl::queue queue{sycl::property::queue::enable_profiling{}};
    int i = 1337;
    sycl::buffer<int, 1> buf{&i, sycl::range<1>(1)};
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.single_task<class basic_single_task>([=]() { acc[0] = 42; });
    });
    auto evt = queue.submit([&](sycl::handler &cgh) {
      cgh.update_host(buf.get_access<sycl::access::mode::read>(cgh));
    });
    BOOST_WARN_NO_THROW(evt.get_profiling_info<
                        sycl::info::event_profiling::command_start>());
  }
}

// Profiled durations must be consistent with host wall-clock time, e.g. device
// timestamp ticks must be converted to nanoseconds.
BOOST_AUTO_TEST_CASE(queue_profiling_duration_vs_wall_clock) {
  if (default_device_supports_profiling()) {
    sycl::queue queue{
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}}};
    constexpr size_t n = 4 * 1024 * 1024;
    float *data = sycl::malloc_device<float>(n, queue);
    auto launch = [&]() {
      return queue.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
        float x = idx[0];
        for (int r = 0; r < 256; ++r)
          x = x * 1.0001f + 0.5f;
        data[idx] = x;
      });
    };
    launch().wait(); // warmup

    // Measure several back-to-back launches so that the profiled span from
    // the first start to the last end is dominated by device execution and
    // comparable to host wall-clock time.
    constexpr int num_launches = 10;
    std::vector<sycl::event> events;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_launches; ++i)
      events.push_back(launch());
    queue.wait();
    const double wall_ns = std::chrono::duration<double, std::nano>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    const double device_ns = static_cast<double>(
        events.back()
            .get_profiling_info<sycl::info::event_profiling::command_end>() -
        events.front()
            .get_profiling_info<sycl::info::event_profiling::command_start>());

    // The device profiling time excludes submission of the first launch, so it
    // may be somewhat shorter than wall-clock time, but a wrong timestamp unit
    // (e.g. device ticks instead of nanoseconds) shows up as a large factor.
    BOOST_CHECK_LE(device_ns, wall_ns + 1e6);
    BOOST_CHECK_GE(device_ns * 2, wall_ns);
    sycl::free(data, queue);
  }
}

// Profiled operations whose events are dropped immediately must not release
// backend profiling resources while the device is still using them.
BOOST_AUTO_TEST_CASE(queue_profiling_dropped_events) {
  if (default_device_supports_profiling()) {
    sycl::queue queue{
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}}};
    constexpr size_t n = 1 << 16;
    int *data = sycl::malloc_device<int>(n, queue);
    queue.memset(data, 0, n * sizeof(int));
    for (int i = 0; i < 500; ++i)
      queue.parallel_for(sycl::range<1>{n},
                         [=](sycl::id<1> idx) { data[idx] += 1; });
    queue.wait();

    int result = 0;
    queue.memcpy(&result, data + n - 1, sizeof(int)).wait();
    BOOST_CHECK_EQUAL(result, 500);
    sycl::free(data, queue);
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
