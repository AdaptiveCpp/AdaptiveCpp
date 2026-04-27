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

BOOST_FIXTURE_TEST_SUITE(usm_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(device_allocation_functions) {
  // Basic check that allocations work
  sycl::queue q;

  std::size_t count = 1024;

  int *device_mem_ptr = sycl::malloc_device<int>(count, q);
  int *aligned_device_mem_ptr =
      sycl::aligned_alloc_device<int>(sizeof(int), count, q);
  std::vector<int> unregistered_data(100);

  BOOST_TEST(device_mem_ptr != nullptr);
  BOOST_TEST(aligned_device_mem_ptr != nullptr);

  auto verify_allocation_type = [&](void *ptr, sycl::usm::alloc expected) {
    sycl::usm::alloc type = sycl::get_pointer_type(ptr, q.get_context());
    BOOST_CHECK(type == expected);
  };

  if (q.get_context().is_host()) {
    verify_allocation_type(device_mem_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_device_mem_ptr, sycl::usm::alloc::host);
    verify_allocation_type(unregistered_data.data(), sycl::usm::alloc::host);
  }
  else {
    verify_allocation_type(device_mem_ptr, sycl::usm::alloc::device);
    verify_allocation_type(aligned_device_mem_ptr, sycl::usm::alloc::device);
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

  sycl::free(device_mem_ptr, q);
  sycl::free(aligned_device_mem_ptr, q);
}

BOOST_AUTO_TEST_CASE(host_allocation_functions) {
  // Basic check that allocations work
  sycl::queue q;
  if (!q.get_device().has(sycl::aspect::usm_host_allocations)) {
    return;
  }

  std::size_t count = 1024;

  int *host_ptr = sycl::malloc_host<int>(count, q);
  int *aligned_host_ptr =
      sycl::aligned_alloc_host<int>(sizeof(int), count, q);

  BOOST_TEST(host_ptr != nullptr);
  BOOST_TEST(aligned_host_ptr != nullptr);

  auto verify_allocation_type = [&](void *ptr, sycl::usm::alloc expected) {
    sycl::usm::alloc type = sycl::get_pointer_type(ptr, q.get_context());
    BOOST_CHECK(type == expected);
  };

  verify_allocation_type(host_ptr, sycl::usm::alloc::host);
  verify_allocation_type(aligned_host_ptr, sycl::usm::alloc::host);

  auto verify_device = [&](void *ptr) {
    // TODO: For a more robust testing if we actually
    // have multiple devices available, we should perform
    // allocations on multiple devices and check that
    // they are all retrieved correctly, instead of
    // just working on a default queue
    sycl::device dev = sycl::get_pointer_device(ptr, q.get_context());
    BOOST_CHECK(dev == q.get_device());
  };

  verify_device(host_ptr);
  verify_device(aligned_host_ptr);

  sycl::free(host_ptr, q);
  sycl::free(aligned_host_ptr, q);
}

BOOST_AUTO_TEST_CASE(shared_allocation_functions) {
  // Basic check that allocations work
  sycl::queue q;
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return;
  }

  std::size_t count = 1024;

  int *shared_ptr = sycl::malloc_shared<int>(count, q);
  int *aligned_shared_ptr =
      sycl::aligned_alloc_shared<int>(sizeof(int), count, q);

  BOOST_TEST(shared_ptr != nullptr);
  BOOST_TEST(aligned_shared_ptr != nullptr);

  auto verify_allocation_type = [&](void *ptr, sycl::usm::alloc expected) {
    sycl::usm::alloc type = sycl::get_pointer_type(ptr, q.get_context());
    BOOST_CHECK(type == expected);
  };

  if (q.get_context().is_host()) {
    verify_allocation_type(shared_ptr, sycl::usm::alloc::host);
    verify_allocation_type(aligned_shared_ptr, sycl::usm::alloc::host);
  }
  else {
    // As of yet, ROCm does not have proper shared allocations
    // and gives us device-accessible host memory instead.
    if(q.get_device().get_backend() != sycl::backend::hip) {
      verify_allocation_type(shared_ptr, sycl::usm::alloc::shared);
      verify_allocation_type(aligned_shared_ptr, sycl::usm::alloc::shared);
    }
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

  verify_device(shared_ptr);
  verify_device(aligned_shared_ptr);

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

  q.wait();
}

BOOST_AUTO_TEST_CASE(allocations_in_kernels) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};
  bool host_usm_support = q.get_device().has(sycl::aspect::usm_host_allocations);
  bool shared_usm_support = q.get_device().has(sycl::aspect::usm_shared_allocations);

  std::size_t test_size = 4096;
  int *shared_allocation = shared_usm_support ? sycl::malloc_shared<int>(test_size, q) : nullptr;
  int *explicit_allocation = sycl::malloc_device<int>(test_size, q);
  int *mapped_host_allocation = host_usm_support ? sycl::malloc_host<int>(test_size, q) : nullptr;

  q.single_task<class usm_alloc_single_task>([=]() {
    for (int i = 0; i < test_size; ++i) {
      if (shared_allocation)
        shared_allocation[i] = i;
      explicit_allocation[i] = i;
      if (mapped_host_allocation)
        mapped_host_allocation[i] = i;
    }
  });

  q.parallel_for<class usm_alloc_pf>(sycl::range<1>{test_size},
                                     [=] (sycl::id<1> idx) {
                                       // Use idx directly to also make sure
                                       // that implicit conversion to size_t
                                       // works
                                       if (shared_allocation)
                                         shared_allocation[idx] += 1;
                                       explicit_allocation[idx] += 1;
                                       if (mapped_host_allocation)
                                         mapped_host_allocation[idx] += 1;
                                     });

  q.parallel_for<class usm_alloc_pf2>(sycl::range<1>{test_size},
                                     [=] (sycl::item<1> idx) {
                                       // Use item directly to also make sure
                                       // that implicit conversion to size_t
                                       // works
                                       if (shared_allocation)
                                         shared_allocation[idx] += 1;
                                       explicit_allocation[idx] += 1;
                                       if (mapped_host_allocation)
                                         mapped_host_allocation[idx] += 1;
                                     });

  q.parallel_for<class usm_alloc_ndrange_pf>(
      sycl::nd_range<1>{sycl::range<1>{test_size}, sycl::range<1>{128}},
      [=](sycl::nd_item<1> idx) {
        if (shared_allocation)
          shared_allocation[idx.get_global_id(0)] += 1;
        explicit_allocation[idx.get_global_id(0)] += 1;
        if (mapped_host_allocation)
          mapped_host_allocation[idx.get_global_id(0)] += 1;
      });

  std::vector<int> host_explicit_allocation(test_size);
  q.memcpy(host_explicit_allocation.data(), explicit_allocation,
           test_size * sizeof(int));
  q.wait();

  for (int i = 0; i < test_size; ++i){
    if (shared_allocation)
      BOOST_TEST(shared_allocation[i] == i + 3);
    BOOST_TEST(host_explicit_allocation[i] == i + 3);
    if (mapped_host_allocation)
      BOOST_TEST(mapped_host_allocation[i] == i + 3);
  }

  if (shared_allocation)
    sycl::free(shared_allocation, q);
  sycl::free(explicit_allocation, q);
  if (mapped_host_allocation)
    sycl::free(mapped_host_allocation, q);
}

BOOST_AUTO_TEST_CASE(memcpy) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};
  sycl::queue ooo_q;

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
  if (q.get_device().has(sycl::aspect::usm_shared_allocations))
  {
    int *shared_mem = sycl::malloc_shared<int>(test_size, q);
    test_device_host_copies(shared_mem);
    sycl::free(shared_mem, q);
  }

  // memcpy device->shared
  // memcpy shared->device
  if (q.get_device().has(sycl::aspect::usm_shared_allocations))
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
  if (q.get_device().has(sycl::aspect::usm_host_allocations))
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

  if (q.get_device().has(sycl::aspect::usm_shared_allocations))
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
  // memcpy host->host, out-of-order queue
  if (q.get_device().has(sycl::aspect::usm_host_allocations))
  {
    int *mem = sycl::malloc_host<int>(test_size, ooo_q);
    int *mem2 = sycl::malloc_host<int>(test_size, ooo_q);

    for (std::size_t i = 0; i < test_size; ++i)
      mem[i] = initial_data[i];

    q.memcpy(mem2, mem, sizeof(int) * test_size);
    q.wait();

    for (std::size_t i = 0; i < test_size; ++i)
      BOOST_TEST(mem2[i] == initial_data[i]);

    sycl::free(mem, ooo_q);
    sycl::free(mem2, ooo_q);
  }
}

BOOST_AUTO_TEST_CASE(usm_fill) {
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

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
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
      return;

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

BOOST_AUTO_TEST_CASE(allocation_zero_bytes) {
  // SYCL standard requires zero-byte allocations to be handled gracefully
  // We just check that no errors are thrown
  sycl::queue q;

  bool host_support = q.get_device().has(sycl::aspect::usm_host_allocations);
  bool shared_support = q.get_device().has(sycl::aspect::usm_shared_allocations);

  int *device_mem_ptr = sycl::malloc_device<int>(0, q);
  if (device_mem_ptr)
    sycl::free(device_mem_ptr, q);
  int *aligned_device_mem_ptr =
      sycl::aligned_alloc_device<int>(sizeof(int), 0, q);
  if (aligned_device_mem_ptr)
    sycl::free(aligned_device_mem_ptr, q);

  int *host_ptr = host_support ? sycl::malloc_host<int>(0, q) : nullptr;
  if (host_ptr)
    sycl::free(host_ptr, q);
  int *aligned_host_ptr = host_support ?
      sycl::aligned_alloc_host<int>(sizeof(int), 0, q) : nullptr;
  if (aligned_host_ptr)
    sycl::free(aligned_host_ptr, q);

  int *shared_ptr = shared_support ? sycl::malloc_shared<int>(0, q) : nullptr;
  if (shared_ptr)
    sycl::free(shared_ptr, q);
  int *aligned_shared_ptr = shared_support ?
      sycl::aligned_alloc_shared<int>(sizeof(int), 0, q) : nullptr;
  if (aligned_shared_ptr)
    sycl::free(aligned_shared_ptr, q);
}

namespace linked_list {

struct Node {
  int id;
  Node *next;
};

std::pair<int, int>
enqueue_kernel(int max_iterations, Node* start, sycl::queue q) {
  int *output_id = sycl::malloc_device<int>(1, q);
  int *output_iterations = sycl::malloc_device<int>(1, q);

  // Set an iteration limit to avoid an infinite loop in the case
  // of erroneous behavior.
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      Node *curr = start;
      int iterations = 0;
      while(iterations < max_iterations && curr)
      {
        *output_id = curr->id;
        ++iterations;
        curr = curr->next;
      }
      *output_iterations = iterations;
    });
  });

  int result_id, result_iterations;
  q.memcpy(&result_id, output_id, sizeof(int));
  q.memcpy(&result_iterations, output_iterations, sizeof(int));
  q.wait();

  sycl::free(output_id, q);
  sycl::free(output_iterations, q);

  return std::make_pair(result_id, result_iterations);
}
} // namespace linked_list

BOOST_AUTO_TEST_CASE(linked_list_single_alloc) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::metal) {
    BOOST_TEST_MESSAGE("Not yet supported on Metal backend");
    return;
  }

  const int num_nodes = 3;
  linked_list::Node *nodes = sycl::malloc_device<linked_list::Node>(num_nodes, q);
  linked_list::Node nodeC{2, nullptr};
  linked_list::Node nodeB{1, nodes + 2};
  linked_list::Node nodeA{0, nodes + 1};

  q.memcpy(nodes, &nodeA, sizeof(linked_list::Node));
  q.memcpy(nodes + 1, &nodeB, sizeof(linked_list::Node));
  q.memcpy(nodes + 2, &nodeC, sizeof(linked_list::Node));
  q.wait();

  const int max_iterations = num_nodes + 1;
  auto [result_id, result_iterations] = linked_list::enqueue_kernel(max_iterations, nodes, q);

  BOOST_CHECK(2 == result_id);
  BOOST_CHECK(num_nodes == result_iterations);

  sycl::free(nodes, q);
}

BOOST_AUTO_TEST_CASE(linked_list_separate_alloc) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::metal) {
    BOOST_TEST_MESSAGE("Not yet supported on Metal backend");
    return;
  }

  constexpr int num_nodes = 3;
  linked_list::Node* nodes[num_nodes];
  for (int i = 0; i < num_nodes; i++) {
    nodes[i] = sycl::malloc_device<linked_list::Node>(1, q);
  }

  for (int i = 0; i < num_nodes; i++) {
    linked_list::Node* next = (i == num_nodes - 1) ? nullptr : nodes[i+1];
    linked_list::Node node{i, next};
    q.memcpy(nodes[i], &node, sizeof(linked_list::Node)).wait();
  }

  const int max_iterations = num_nodes+1;
  auto [result_id, result_iterations] = linked_list::enqueue_kernel(max_iterations, nodes[0], q);

  BOOST_CHECK(2 == result_id);
  BOOST_CHECK(num_nodes == result_iterations);

  for (int i = 0; i < num_nodes; i++) {
    sycl::free(nodes[i], q);
  }
}

BOOST_AUTO_TEST_CASE(usm_shared_ptr_gpu_delta_constant) {
  sycl::queue q{sycl::property::queue::in_order{}};

  if (!q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return;
  }

  static constexpr int K = 8;
  static constexpr int sizes[K] = {4096, 8192, 1, 16384, 512, 32768, 2048, 65536};

  int* a[K];
  for (int k = 0; k < K; ++k)
    a[k] = sycl::malloc_shared<int>(sizes[k], q);

  uint64_t* gpu_addrs = sycl::malloc_shared<uint64_t>(K, q);

  int *p0=a[0], *p1=a[1], *p2=a[2], *p3=a[3],
      *p4=a[4], *p5=a[5], *p6=a[6], *p7=a[7];

  q.single_task([=]() {
    gpu_addrs[0] = reinterpret_cast<uint64_t>(p0);
    gpu_addrs[1] = reinterpret_cast<uint64_t>(p1);
    gpu_addrs[2] = reinterpret_cast<uint64_t>(p2);
    gpu_addrs[3] = reinterpret_cast<uint64_t>(p3);
    gpu_addrs[4] = reinterpret_cast<uint64_t>(p4);
    gpu_addrs[5] = reinterpret_cast<uint64_t>(p5);
    gpu_addrs[6] = reinterpret_cast<uint64_t>(p6);
    gpu_addrs[7] = reinterpret_cast<uint64_t>(p7);
  });
  q.wait();

  int64_t first_delta = static_cast<int64_t>(gpu_addrs[0] - reinterpret_cast<uint64_t>(a[0]));
  bool all_same = true;
  for (int k = 1; k < K; ++k) {
    int64_t delta = static_cast<int64_t>(gpu_addrs[k] - reinterpret_cast<uint64_t>(a[k]));
    if (delta != first_delta) {
      all_same = false;
      BOOST_TEST_MESSAGE("delta mismatch at k=" << k
        << " expected=" << first_delta << " got=" << delta);
    }
  }
  BOOST_CHECK(all_same);

  for (int k = 0; k < K; ++k) sycl::free(a[k], q);
  sycl::free(gpu_addrs, q);
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
