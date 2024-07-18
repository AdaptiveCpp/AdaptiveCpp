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

#include <exception>
#include <vector>

#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/device_selector.hpp"
#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

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
  std::vector<int> unregistered_data(100);
  
  
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
    verify_allocation_type(unregistered_data.data(), sycl::usm::alloc::host);
  }
  else {
    verify_allocation_type(device_mem_ptr, sycl::usm::alloc::device);
    verify_allocation_type(aligned_device_mem_ptr, sycl::usm::alloc::device);
    verify_allocation_type(host_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_host_ptr, sycl::usm::alloc::host);
    // As of yet, ROCm does not have proper shared allocations
    // and gives us device-accessible host memory instead.
    if(q.get_device().get_backend() != sycl::backend::hip) {
      verify_allocation_type(shared_ptr, sycl::usm::alloc::shared);
      verify_allocation_type(aligned_shared_ptr, sycl::usm::alloc::shared);
    }
    verify_allocation_type(unregistered_data.data(), sycl::usm::alloc::unknown);
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
  sycl::queue q{sycl::property_list{
      sycl::property::queue::in_order{},
      sycl::property::queue::AdaptiveCpp_retargetable{} // Needed for accurate
                                                        // get_wait_list results
  }};

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

BOOST_AUTO_TEST_CASE(allocations_in_kernels) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  int *shared_allocation = sycl::malloc_shared<int>(test_size, q);
  int *explicit_allocation = sycl::malloc_device<int>(test_size, q);
  int *mapped_host_allocation = sycl::malloc_host<int>(test_size, q);
  
  q.single_task<class usm_alloc_single_task>([=]() {
    for (int i = 0; i < test_size; ++i) {
      shared_allocation[i] = i;
      explicit_allocation[i] = i;
      mapped_host_allocation[i] = i;
    }
  });

  q.parallel_for<class usm_alloc_pf>(sycl::range<1>{test_size},
                                     [=] (sycl::id<1> idx) {
                                       // Use idx directly to also make sure
                                       // that implicit conversion to size_t
                                       // works
                                       shared_allocation[idx] += 1;
                                       explicit_allocation[idx] += 1;
                                       mapped_host_allocation[idx] += 1;
                                     });

  q.parallel_for<class usm_alloc_pf2>(sycl::range<1>{test_size},
                                     [=] (sycl::item<1> idx) {
                                       // Use item directly to also make sure
                                       // that implicit conversion to size_t
                                       // works
                                       shared_allocation[idx] += 1;
                                       explicit_allocation[idx] += 1;
                                       mapped_host_allocation[idx] += 1;
                                     });

  q.parallel_for<class usm_alloc_ndrange_pf>(
      sycl::nd_range<1>{sycl::range<1>{test_size}, sycl::range<1>{128}},
      [=](sycl::nd_item<1> idx) {
        shared_allocation[idx.get_global_id(0)] += 1;
        explicit_allocation[idx.get_global_id(0)] += 1;
        mapped_host_allocation[idx.get_global_id(0)] += 1;
      });

  std::vector<int> host_explicit_allocation(test_size);
  q.memcpy(host_explicit_allocation.data(), explicit_allocation,
           test_size * sizeof(int));
  q.wait();

  for (int i = 0; i < test_size; ++i){
    BOOST_TEST(shared_allocation[i] == i + 3);
    BOOST_TEST(host_explicit_allocation[i] == i + 3);
    BOOST_TEST(mapped_host_allocation[i] == i + 3);
  }

  sycl::free(shared_allocation, q);
  sycl::free(explicit_allocation, q);
  sycl::free(mapped_host_allocation, q);
}
BOOST_AUTO_TEST_CASE(memcpy) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  std::vector<int> initial_data(test_size);

  for (std::size_t i = 0; i < initial_data.size(); ++i)
    initial_data[i] = i;

  auto test_device_host_copies = [&](int *dev_ptr) {
    std::vector<int> host_data(test_size);
    q.memcpy(dev_ptr, initial_data.data(), sizeof(int) * test_size);
    q.memcpy(host_data.data(), dev_ptr, sizeof(int) * test_size);

    q.wait();

    for (std::size_t i = 0; i < test_size; ++i) {
      BOOST_TEST(host_data[i] == initial_data[i]);
    }
  };


  // memcpy host->explicit device
  // memcpy explicit device->host
  {
    int *device_mem = sycl::malloc_device<int>(test_size, q);
    test_device_host_copies(device_mem);
    sycl::free(device_mem, q);
  }
  // memcpy host->shared
  // memcpy shared->host
  {
    int *shared_mem = sycl::malloc_shared<int>(test_size, q);
    test_device_host_copies(shared_mem);
    sycl::free(shared_mem, q);
  }

  // memcpy device->shared
  // memcpy shared->device
  {
    int *device_mem = sycl::malloc_device<int>(test_size, q);
    int *shared_mem = sycl::malloc_shared<int>(test_size, q);

    q.memcpy(device_mem, initial_data.data(), sizeof(int) * test_size);
    q.memcpy(shared_mem, device_mem, sizeof(int) * test_size);

    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(shared_mem[i] == initial_data[i]);

    int *device_mem2 = sycl::malloc_device<int>(test_size, q);
    std::vector<int> host_data(test_size);

    q.memcpy(device_mem2, shared_mem, sizeof(int) * test_size);
    q.memcpy(host_data.data(), device_mem2, sizeof(int) * test_size);

    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(host_data[i] == initial_data[i]);
    
    sycl::free(device_mem, q);
    sycl::free(device_mem2, q);
    sycl::free(shared_mem, q);
  }

  // memcpy host->host
  {
    int *host_mem = sycl::malloc_host<int>(test_size, q);
    int *host_mem2 = sycl::malloc_host<int>(test_size, q);

    for (std::size_t i = 0; i < test_size; ++i)
      host_mem[i] = initial_data[i];

    q.memcpy(host_mem2, host_mem, sizeof(int) * test_size);
    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(host_mem2[i] == initial_data[i]);

    sycl::free(host_mem, q);
    sycl::free(host_mem2, q);
  }

  // memcpy device->device
  {
    int *device_mem = sycl::malloc_device<int>(test_size, q);
    int *device_mem2 = sycl::malloc_device<int>(test_size, q);
    std::vector<int> host_data(test_size);
    
    q.memcpy(device_mem, initial_data.data(), test_size * sizeof(int));
    q.memcpy(device_mem2, device_mem, test_size * sizeof(int));
    q.memcpy(host_data.data(), device_mem2, test_size * sizeof(int));
    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(host_data[i] == initial_data[i]);

    sycl::free(device_mem,  q);
    sycl::free(device_mem2, q);
  }
  // memcpy shared->shared
  {
    int *shared_mem = sycl::malloc_shared<int>(test_size, q);
    int *shared_mem2 = sycl::malloc_shared<int>(test_size, q);

    for (std::size_t i = 0; i < test_size; ++i)
      shared_mem[i] = initial_data[i];

    q.memcpy(shared_mem2, shared_mem, sizeof(int) * test_size);
    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(shared_mem2[i] == initial_data[i]);

    sycl::free(shared_mem, q);
    sycl::free(shared_mem2, q);
  }
}
BOOST_AUTO_TEST_CASE(usm_fill) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  int* shared_mem = sycl::malloc_shared<int>(test_size, q);
  for (int i = 0; i < test_size; ++i)
    shared_mem[i] = 0;

  int fill_value = 1234567890;
  q.fill(shared_mem+1, fill_value, test_size-2);
  q.wait();

  for (int i = 0; i < test_size; ++i) {
    if (i == 0 || i == test_size - 1)
      BOOST_TEST(shared_mem[i] == 0);
    else
      BOOST_TEST(shared_mem[i] == fill_value);
  }

  sycl::free(shared_mem, q);
}
BOOST_AUTO_TEST_CASE(memset) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  unsigned char *mem = sycl::malloc_device<unsigned char>(test_size, q);

  q.memset(mem, 0, test_size);
  q.memset(mem + 1, 12, test_size - 2);
  std::vector<unsigned char> host_mem(test_size);
  q.memcpy(host_mem.data(), mem, test_size);

  q.wait();

  for (int i = 0; i < test_size; ++i) {
    if (i == 0 || i == test_size - 1)
      BOOST_TEST(host_mem[i] == 0);
    else
      BOOST_TEST(host_mem[i] == 12);
  }

  sycl::free(mem, q);
}
BOOST_AUTO_TEST_CASE(prefetch) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  int *shared_mem = sycl::malloc_shared<int>(test_size, q);

  for (std::size_t i = 0; i < test_size; ++i)
    shared_mem[i] = i;

  q.prefetch(shared_mem, test_size * sizeof(int));
  q.parallel_for<class usm_prefetch_test_kernel>(
      sycl::range<1>{test_size},
      [=](sycl::id<1> idx) { shared_mem[idx.get(0)] += 1; });
  
  q.wait();

  // Test prefetching to host using a host_queue
  {
    sycl::queue host_queue{q.get_context(), sycl::host_selector{}};
    host_queue.prefetch(shared_mem, test_size * sizeof(int));
    host_queue.wait();
  }
  for (std::size_t i = 0; i < test_size; ++i)
    BOOST_TEST(shared_mem[i] == i + 1);
  
  sycl::free(shared_mem, q);
}
BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
