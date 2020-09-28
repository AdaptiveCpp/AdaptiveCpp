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

using namespace cl;

BOOST_FIXTURE_TEST_SUITE(usm_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(allocation_functions) {
  // Basic check that allocations work
  sycl::queue q;

  std::size_t count = 1024;

  int *device_mem_ptr = sycl::malloc_device<int>(count, q);
  int *aligned_device_mem_ptr =
      sycl::aligned_alloc_device<int>(sizeof(int), count, q);
  int *host_ptr = sycl::malloc_host<int>(count, q);
  int *aligned_host_ptr =
      sycl::aligned_alloc_host<int>(sizeof(int), count, q);
  int *shared_ptr = sycl::malloc_shared<int>(count, q);
  int *aligned_shared_ptr =
      sycl::aligned_alloc_shared<int>(sizeof(int), count, q);

  BOOST_TEST(device_mem_ptr != nullptr);
  BOOST_TEST(aligned_device_mem_ptr != nullptr);
  BOOST_TEST(host_ptr != nullptr);
  BOOST_TEST(aligned_host_ptr != nullptr);
  BOOST_TEST(shared_ptr != nullptr);
  BOOST_TEST(aligned_shared_ptr != nullptr);

  auto verify_allocation_type = [&](void *ptr, sycl::usm::alloc expected) {
    sycl::usm::alloc type = sycl::get_pointer_type(ptr, q.get_context());
    BOOST_CHECK(type == expected);
  };

  if (q.get_context().is_host()) {
    verify_allocation_type(device_mem_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_device_mem_ptr, sycl::usm::alloc::host);
    verify_allocation_type(host_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_host_ptr, sycl::usm::alloc::host);
    verify_allocation_type(shared_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_shared_ptr, sycl::usm::alloc::host);
  }
  else {
    verify_allocation_type(device_mem_ptr, sycl::usm::alloc::device);
    verify_allocation_type(aligned_device_mem_ptr, sycl::usm::alloc::device);
    verify_allocation_type(host_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_host_ptr, sycl::usm::alloc::host);
    verify_allocation_type(shared_ptr, sycl::usm::alloc::shared);
    verify_allocation_type(aligned_shared_ptr, sycl::usm::alloc::shared);
  }

  auto verify_device = [&](void *ptr) {
    // TODO: For a more robust testing if we actually
    // have multiple devices available, we should perform
    // allocations on multiple devices and check that
    // they are all retrieved correctly, instead of
    // just working on a default queue
    sycl::device dev = sycl::get_pointer_device(ptr, q.get_context());
    BOOST_CHECK(dev == q.get_device());
  };

  verify_device(device_mem_ptr);
  verify_device(aligned_device_mem_ptr);
  verify_device(host_ptr);
  verify_device(aligned_host_ptr);
  verify_device(shared_ptr);
  verify_device(aligned_shared_ptr);
  
  
  sycl::free(device_mem_ptr, q);
  sycl::free(aligned_device_mem_ptr, q);
  sycl::free(host_ptr, q);
  sycl::free(aligned_host_ptr, q);
  sycl::free(shared_ptr, q);
  sycl::free(aligned_shared_ptr, q);

}

BOOST_AUTO_TEST_CASE(explicit_queue_dependencies) {
  sycl::queue q;

  // By default, we should have an out-of-order queue
  BOOST_CHECK(!q.is_in_order());

  // Make sure that there are no dependencies between tasks
  // by default
  sycl::event evt1 = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class Queue_deps_kernel1>([](){});
  });

  BOOST_CHECK(evt1.get_wait_list().empty());

  sycl::event evt2 = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class Queue_deps_kernel2>([](){});
  });

  BOOST_CHECK(evt2.get_wait_list().empty());

  // Make sure that we depend on previous tasks once we use
  // depends_on()
  sycl::event evt3 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(evt2);
    cgh.single_task<class Queue_deps_kernel3>([](){});
  });

  BOOST_CHECK(evt3.get_wait_list().size() == 1);
  BOOST_CHECK(evt3.get_wait_list()[0] == evt2);

  sycl::event evt4 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(evt3);
    cgh.single_task<class Queue_deps_kernel4>([](){});
  });

  BOOST_CHECK(evt4.get_wait_list().size() == 1);
  BOOST_CHECK(evt4.get_wait_list()[0] == evt3);
}


BOOST_AUTO_TEST_CASE(in_order_queue) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  BOOST_CHECK(q.is_in_order());

  sycl::event evt1 = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class In_order_queue_kernel1>([](){});
  });

  BOOST_CHECK(evt1.get_wait_list().empty());

  sycl::event evt2 = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class In_order_queue_kernel2>([](){});
  });

  BOOST_CHECK(evt2.get_wait_list().size() == 1);
  BOOST_CHECK(evt2.get_wait_list()[0] == evt1);

  sycl::event evt3 = q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class In_order_queue_kernel3>([](){});
  });

  BOOST_CHECK(evt3.get_wait_list().size() == 1);
  BOOST_CHECK(evt3.get_wait_list()[0] == evt2);
}



BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
