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

BOOST_AUTO_TEST_CASE(invalid_pointer_argument) {
  sycl::queue q;

  void** data_in = sycl::malloc_device<void*>(1, q);
  void** data_out = sycl::malloc_device<void*>(1, q);

  void* data = reinterpret_cast<void*>(0x1);
  q.memcpy(data_in, &data, sizeof(void*)).wait();

  q.single_task([=](){
    *data_out = *data_in;
  }).wait();

  void* data_result = nullptr;
  q.memcpy(&data_result, data_out, sizeof(void*)).wait();
  BOOST_CHECK(data_result == reinterpret_cast<void*>(0x1));

  sycl::free(data_in, q);
  sycl::free(data_out, q);

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

  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
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

  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
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

namespace {
namespace shared_indirection {

struct ListNode {
  int id;
  ListNode *next;
};

struct TreeNode {
  int id;
  TreeNode *left;
  TreeNode *right;
};

struct RotateNode {
  int key;
  RotateNode *left;
  RotateNode *right;
};

struct ColVal {
  int col;
  float val;
};

struct RowSlice {
  ColVal *data;
  int len;
};

constexpr int TrieAlpha = 26;

struct TrieNode {
  TrieNode *children[TrieAlpha];
  bool is_end;
};

bool trie_contains(TrieNode *root, const char *word, int len) {
  TrieNode *cur = root;
  for (int i = 0; i < len; ++i) {
    int c = word[i] - 'a';
    if (c < 0 || c >= TrieAlpha) {
      return false;
    }
    cur = cur->children[c];
    if (!cur) {
      return false;
    }
  }
  return cur->is_end;
}

struct GraphNode {
  int id;
  int num_neighbors;
  GraphNode **neighbors;
};

int bfs_reachable(GraphNode *src) {
  constexpr int MaxV = 16;
  bool visited[MaxV] = {};
  GraphNode *queue[MaxV];
  int head = 0, tail = 0;

  visited[src->id] = true;
  queue[tail++] = src;

  while (head < tail) {
    GraphNode *cur = queue[head++];
    for (int i = 0; i < cur->num_neighbors; ++i) {
      GraphNode *nb = cur->neighbors[i];
      if (!visited[nb->id]) {
        visited[nb->id] = true;
        queue[tail++] = nb;
      }
    }
  }
  return tail;
}

constexpr int BTreeT = 2;
constexpr int BTreeMaxKeys = 2 * BTreeT - 1;
constexpr int BTreeMaxChildren = 2 * BTreeT;

struct BTreeNode {
  int keys[BTreeMaxKeys];
  BTreeNode *children[BTreeMaxChildren];
  int num_keys;
  bool leaf;
};

int btree_search(BTreeNode *root, int key) {
  BTreeNode *cur = root;
  while (cur) {
    int i = 0;
    while (i < cur->num_keys && key > cur->keys[i]) {
      ++i;
    }
    if (i < cur->num_keys && cur->keys[i] == key) {
      return 1;
    }
    if (cur->leaf) {
      return 0;
    }
    cur = cur->children[i];
  }
  return 0;
}

} // namespace shared_indirection
} // namespace

BOOST_AUTO_TEST_CASE(shared_double_indirection) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  static constexpr int N = 16;
  static constexpr std::size_t PtrBytes = N * sizeof(int *);
  static constexpr std::size_t ValBytes = N * sizeof(int);

  char *base = sycl::malloc_shared<char>(PtrBytes + ValBytes, q);
  int **ptrs = reinterpret_cast<int **>(base);
  int *values = reinterpret_cast<int *>(base + PtrBytes);
  int *results = sycl::malloc_shared<int>(N, q);

  for (int i = 0; i < N; ++i) {
    values[i] = (i + 1) * 10;
    ptrs[i] = &values[i];
    results[i] = 0;
  }

  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
    int i = static_cast<int>(idx[0]);
    results[i] = *ptrs[i];
  });
  q.wait();

  for (int i = 0; i < N; ++i)
    BOOST_CHECK(results[i] == (i + 1) * 10);

  sycl::free(base, q);
  sycl::free(results, q);
}

BOOST_AUTO_TEST_CASE(shared_trie_search) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using namespace shared_indirection;
  static constexpr int MaxNodes = 32;
  static constexpr int NumQueries = 8;
  static constexpr int WordLen = 8;

  struct Query {
    char word[WordLen];
    int len;
    int expected;
  };

  TrieNode *nodes = sycl::malloc_shared<TrieNode>(MaxNodes, q);
  for (int i = 0; i < MaxNodes; ++i) {
    for (int c = 0; c < TrieAlpha; ++c)
      nodes[i].children[c] = nullptr;
    nodes[i].is_end = false;
  }

  int next = 0;
  TrieNode *root = &nodes[next++];
  auto insert = [&](const char *word) {
    TrieNode *cur = root;
    for (int i = 0; word[i]; ++i) {
      int c = word[i] - 'a';
      if (!cur->children[c])
        cur->children[c] = &nodes[next++];
      cur = cur->children[c];
    }
    cur->is_end = true;
  };

  insert("cat");
  insert("car");
  insert("card");
  insert("care");
  insert("dog");
  insert("do");

  Query host_queries[NumQueries] = {
    {"cat", 3, 1},  {"car", 3, 1}, {"card", 4, 1}, {"care", 4, 1},
    {"dog", 3, 1},  {"do", 2, 1},  {"ca", 2, 0},   {"door", 4, 0}};

  Query *queries = sycl::malloc_shared<Query>(NumQueries, q);
  int *results = sycl::malloc_shared<int>(NumQueries, q);
  for (int i = 0; i < NumQueries; ++i) {
    queries[i] = host_queries[i];
    results[i] = 0;
  }

  q.parallel_for(sycl::range<1>(NumQueries), [=](sycl::id<1> idx) {
    int i = static_cast<int>(idx[0]);
    results[i] = trie_contains(root, queries[i].word, queries[i].len) ? 1 : 0;
  });
  q.wait();

  for (int i = 0; i < NumQueries; ++i) {
    BOOST_CHECK(results[i] == host_queries[i].expected);
  }

  sycl::free(nodes, q);
  sycl::free(queries, q);
  sycl::free(results, q);
}

BOOST_AUTO_TEST_CASE(shared_graph_bfs) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using namespace shared_indirection;
  static constexpr int V = 6;
  static constexpr int PtrSlots = 7;
  std::size_t node_bytes = V * sizeof(GraphNode);
  std::size_t ptr_bytes = PtrSlots * sizeof(GraphNode *);

  char *block = sycl::malloc_shared<char>(node_bytes + ptr_bytes, q);
  GraphNode *nodes = reinterpret_cast<GraphNode *>(block);
  GraphNode **ptr_pool = reinterpret_cast<GraphNode **>(block + node_bytes);

  for (int i = 0; i < V; ++i) {
    nodes[i].id = i;
    nodes[i].num_neighbors = 0;
    nodes[i].neighbors = nullptr;
  }
  for (int i = 0; i < PtrSlots; ++i) {
    ptr_pool[i] = nullptr;
  }

  nodes[0].num_neighbors = 2;
  nodes[0].neighbors = &ptr_pool[0];
  ptr_pool[0] = &nodes[1];
  ptr_pool[1] = &nodes[2];
  nodes[1].num_neighbors = 1;
  nodes[1].neighbors = &ptr_pool[2];
  ptr_pool[2] = &nodes[3];
  nodes[2].num_neighbors = 2;
  nodes[2].neighbors = &ptr_pool[3];
  ptr_pool[3] = &nodes[3];
  ptr_pool[4] = &nodes[4];
  nodes[3].num_neighbors = 1;
  nodes[3].neighbors = &ptr_pool[5];
  ptr_pool[5] = &nodes[5];
  nodes[4].num_neighbors = 1;
  nodes[4].neighbors = &ptr_pool[6];
  ptr_pool[6] = &nodes[5];

  int *results = sycl::malloc_shared<int>(V, q);
  for (int i = 0; i < V; ++i)
    results[i] = 0;

  q.parallel_for(sycl::range<1>(V), [=](sycl::id<1> idx) {
    int i = static_cast<int>(idx[0]);
    results[i] = bfs_reachable(&nodes[i]);
  });
  q.wait();

  static constexpr int expected[V] = {6, 3, 4, 2, 2, 1};
  for (int i = 0; i < V; ++i) {
    BOOST_CHECK(results[i] == expected[i]);
  }

  sycl::free(block, q);
  sycl::free(results, q);
}

BOOST_AUTO_TEST_CASE(shared_sparse_matvec) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using namespace shared_indirection;
  static constexpr int R = 4;
  static constexpr int NNZ = 8;
  std::size_t row_bytes = R * sizeof(RowSlice);
  std::size_t cv_bytes = NNZ * sizeof(ColVal);

  char *block = sycl::malloc_shared<char>(row_bytes + cv_bytes, q);
  RowSlice *rows = reinterpret_cast<RowSlice *>(block);
  ColVal *pool = reinterpret_cast<ColVal *>(block + row_bytes);

  pool[0] = {0, 1.0f};
  pool[1] = {2, 2.0f};
  pool[2] = {1, 3.0f};
  pool[3] = {3, 4.0f};
  pool[4] = {0, 5.0f};
  pool[5] = {1, 6.0f};
  pool[6] = {2, 7.0f};
  pool[7] = {3, 8.0f};
  rows[0] = {&pool[0], 2};
  rows[1] = {&pool[2], 2};
  rows[2] = {&pool[4], 3};
  rows[3] = {&pool[7], 1};

  float *x = sycl::malloc_shared<float>(R, q);
  float *y = sycl::malloc_shared<float>(R, q);
  x[0] = 1.f;
  x[1] = 2.f;
  x[2] = 3.f;
  x[3] = 4.f;
  for (int i = 0; i < R; ++i) {
    y[i] = 0.f;
  }

  q.parallel_for(sycl::range<1>(R), [=](sycl::id<1> idx) {
    int i = static_cast<int>(idx[0]);
    ColVal *data = rows[i].data;
    float acc = 0.f;
    for (int k = 0; k < rows[i].len; ++k)
      acc += data[k].val * x[data[k].col];
    y[i] = acc;
  });
  q.wait();

  static constexpr float expected[R] = {7.f, 22.f, 38.f, 32.f};
  for (int i = 0; i < R; ++i) {
    BOOST_CHECK(y[i] == expected[i]);
  }

  sycl::free(block, q);
  sycl::free(x, q);
  sycl::free(y, q);
}

BOOST_AUTO_TEST_CASE(shared_linked_list_append) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using shared_indirection::ListNode;
  static constexpr int NumNodes = 4;
  ListNode *nodes = sycl::malloc_shared<ListNode>(NumNodes + 1, q);

  for (int i = 0; i < NumNodes; ++i) {
    nodes[i].id = i * 10;
    nodes[i].next = (i + 1 < NumNodes) ? &nodes[i + 1] : nullptr;
  }
  nodes[NumNodes].id = NumNodes * 10;
  nodes[NumNodes].next = nullptr;

  ListNode *head = &nodes[0];
  ListNode *new_node = &nodes[NumNodes];
  q.single_task([=]() {
    ListNode *curr = head;
    while (curr->next) {
      curr = curr->next;
    }
    curr->next = new_node;
  });
  q.wait();

  int count = 0;
  int last_id = -1;
  for (ListNode *curr = head; curr && count <= NumNodes + 1; curr = curr->next) {
    last_id = curr->id;
    ++count;
  }
  BOOST_CHECK(count == NumNodes + 1);
  BOOST_CHECK(last_id == NumNodes * 10);

  sycl::free(nodes, q);
}

BOOST_AUTO_TEST_CASE(shared_memcpy_ptr_array) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using shared_indirection::ListNode;
  static constexpr int N = 4;
  ListNode *nodes = sycl::malloc_shared<ListNode>(N, q);
  ListNode **ptrs = sycl::malloc_shared<ListNode *>(N, q);
  int *result = sycl::malloc_shared<int>(1, q);

  for (int i = 0; i < N; ++i) {
    nodes[i].id = i;
    nodes[i].next = (i + 1 < N) ? &nodes[i + 1] : nullptr;
    ptrs[i] = &nodes[i];
  }
  *result = 0;

  q.single_task([=]() {
    ListNode *local_ptrs[N];
    __builtin_memcpy(local_ptrs, ptrs, N * sizeof(ListNode *));

    int sum = nodes[0].id;
    for (int i = 0; i < N; ++i) {
      sum += local_ptrs[i]->id;
    }
    *result = sum;
  });
  q.wait();

  BOOST_CHECK(*result == 6);

  sycl::free(nodes, q);
  sycl::free(ptrs, q);
  sycl::free(result, q);
}

BOOST_AUTO_TEST_CASE(shared_reverse_list) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using shared_indirection::ListNode;
  static constexpr int N = 5;
  ListNode *nodes = sycl::malloc_shared<ListNode>(N, q);
  ListNode **new_head = sycl::malloc_shared<ListNode *>(1, q);
  *new_head = nullptr;

  for (int i = 0; i < N; ++i) {
    nodes[i].id = i;
    nodes[i].next = (i + 1 < N) ? &nodes[i + 1] : nullptr;
  }

  ListNode *head = &nodes[0];
  q.single_task([=]() {
    ListNode *prev = nullptr;
    ListNode *curr = head;
    while (curr) {
      ListNode *next = curr->next;
      curr->next = prev;
      prev = curr;
      curr = next;
    }
    *new_head = prev;
  });
  q.wait();

  int count = 0;
  for (ListNode *curr = *new_head; curr && count < N + 1; curr = curr->next) {
    BOOST_CHECK(curr->id == N - 1 - count);
    ++count;
  }
  BOOST_CHECK(count == N);

  sycl::free(nodes, q);
  sycl::free(new_head, q);
}

BOOST_AUTO_TEST_CASE(shared_ptr_swap) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using shared_indirection::ListNode;
  static constexpr int N = 6;
  ListNode *nodes = sycl::malloc_shared<ListNode>(N, q);

  for (int i = 0; i < 3; ++i) {
    nodes[i].id = i;
    nodes[i].next = (i < 2) ? &nodes[i + 1] : nullptr;
  }
  for (int i = 3; i < 6; ++i) {
    nodes[i].id = i;
    nodes[i].next = (i < 5) ? &nodes[i + 1] : nullptr;
  }

  ListNode *head_a = &nodes[0];
  ListNode *head_b = &nodes[3];
  q.single_task([=]() {
    ListNode *a1 = head_a->next;
    ListNode *b1 = head_b->next;
    ListNode *a2 = a1->next;
    ListNode *b2 = b1->next;
    a1->next = b2;
    b1->next = a2;
  });
  q.wait();

  int ids_a[3];
  int ca = 0;
  for (ListNode *n = head_a; n && ca < 3; n = n->next) {
    ids_a[ca++] = n->id;
  }
  int ids_b[3];
  int cb = 0;
  for (ListNode *n = head_b; n && cb < 3; n = n->next) {
    ids_b[cb++] = n->id;
  }

  BOOST_CHECK(ca == 3);
  BOOST_CHECK(ids_a[0] == 0);
  BOOST_CHECK(ids_a[1] == 1);
  BOOST_CHECK(ids_a[2] == 5);
  BOOST_CHECK(cb == 3);
  BOOST_CHECK(ids_b[0] == 3);
  BOOST_CHECK(ids_b[1] == 4);
  BOOST_CHECK(ids_b[2] == 2);

  sycl::free(nodes, q);
}

BOOST_AUTO_TEST_CASE(shared_binary_tree_traversal) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;
  // TODO: crashes on AMD (HIP) - possibly a compiler issue with local pointer arrays in kernels
  if (q.get_device().get_backend() == sycl::backend::hip)
    return;

  using shared_indirection::TreeNode;
  static constexpr int NumNodes = 7;
  TreeNode *nodes = sycl::malloc_shared<TreeNode>(NumNodes, q);
  int *result = sycl::malloc_shared<int>(1, q);
  *result = 0;

  for (int i = 0; i < NumNodes; ++i) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    nodes[i].id = i;
    nodes[i].left = (left < NumNodes) ? &nodes[left] : nullptr;
    nodes[i].right = (right < NumNodes) ? &nodes[right] : nullptr;
  }

  TreeNode *root = &nodes[0];
  q.single_task([=]() {
    TreeNode *stack[NumNodes];
    int top = 0;
    int sum = 0;
    stack[top++] = root;
    while (top > 0) {
      TreeNode *curr = stack[--top];
      sum += curr->id;
      if (curr->right) {
        stack[top++] = curr->right;
      }
      if (curr->left) {
        stack[top++] = curr->left;
      }
    }
    *result = sum;
  });
  q.wait();

  BOOST_CHECK(*result == NumNodes * (NumNodes - 1) / 2);

  sycl::free(nodes, q);
  sycl::free(result, q);
}

BOOST_AUTO_TEST_CASE(shared_tree_rotate) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using shared_indirection::RotateNode;
  static constexpr int NumNodes = 10;
  RotateNode *nodes = sycl::malloc_shared<RotateNode>(NumNodes, q);
  RotateNode **new_roots = sycl::malloc_shared<RotateNode *>(2, q);

  nodes[0] = {10, &nodes[1], &nodes[2]};
  nodes[1] = {5, &nodes[3], &nodes[4]};
  nodes[2] = {15, nullptr, nullptr};
  nodes[3] = {2, nullptr, nullptr};
  nodes[4] = {7, nullptr, nullptr};
  nodes[5] = {20, &nodes[6], &nodes[7]};
  nodes[6] = {8, &nodes[8], &nodes[9]};
  nodes[7] = {25, nullptr, nullptr};
  nodes[8] = {6, nullptr, nullptr};
  nodes[9] = {9, nullptr, nullptr};
  new_roots[0] = nullptr;
  new_roots[1] = nullptr;

  RotateNode *root0 = &nodes[0];
  RotateNode *root1 = &nodes[5];
  q.parallel_for(sycl::range<1>(2), [=](sycl::id<1> idx) {
    RotateNode *x = idx[0] == 0 ? root0 : root1;
    RotateNode *p = x->left;
    RotateNode *b = p->right;
    x->left = b;
    p->right = x;
    new_roots[idx[0]] = p;
  });
  q.wait();

  RotateNode *r0 = new_roots[0];
  BOOST_CHECK(r0->key == 5);
  BOOST_CHECK(r0->left->key == 2);
  BOOST_CHECK(r0->right->key == 10);
  BOOST_CHECK(r0->right->left->key == 7);
  BOOST_CHECK(r0->right->right->key == 15);

  RotateNode *r1 = new_roots[1];
  BOOST_CHECK(r1->key == 8);
  BOOST_CHECK(r1->left->key == 6);
  BOOST_CHECK(r1->right->key == 20);
  BOOST_CHECK(r1->right->left->key == 9);
  BOOST_CHECK(r1->right->right->key == 25);

  sycl::free(nodes, q);
  sycl::free(new_roots, q);
}

BOOST_AUTO_TEST_CASE(shared_btree_search) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (!q.get_device().has(sycl::aspect::usm_shared_allocations))
    return;

  using namespace shared_indirection;
  static constexpr int NumNodes = 4;
  static constexpr int NumThreads = 1024;
  BTreeNode *nodes = sycl::malloc_shared<BTreeNode>(NumNodes, q);
  int *results = sycl::malloc_shared<int>(NumThreads, q);

  for (int i = 0; i < NumNodes; ++i) {
    for (int k = 0; k < BTreeMaxKeys; ++k) {
      nodes[i].keys[k] = 0;
    }
    for (int c = 0; c < BTreeMaxChildren; ++c) {
      nodes[i].children[c] = nullptr;
    }
    nodes[i].num_keys = 0;
    nodes[i].leaf = true;
  }

  nodes[0].keys[0] = 1;
  nodes[0].keys[1] = 2;
  nodes[0].num_keys = 2;
  nodes[1].keys[0] = 4;
  nodes[1].keys[1] = 5;
  nodes[1].keys[2] = 6;
  nodes[1].num_keys = 3;
  nodes[2].keys[0] = 8;
  nodes[2].keys[1] = 9;
  nodes[2].num_keys = 2;
  nodes[3].keys[0] = 3;
  nodes[3].keys[1] = 7;
  nodes[3].num_keys = 2;
  nodes[3].leaf = false;
  nodes[3].children[0] = &nodes[0];
  nodes[3].children[1] = &nodes[1];
  nodes[3].children[2] = &nodes[2];

  BTreeNode *root = &nodes[3];
  q.parallel_for(sycl::range<1>(NumThreads), [=](sycl::id<1> idx) {
    int key = static_cast<int>(idx[0]);
    results[key] = btree_search(root, key);
  });
  q.wait();

  for (int key = 0; key < NumThreads; ++key) {
    int expected = key >= 1 && key <= 9 ? 1 : 0;
    BOOST_CHECK(results[key] == expected);
  }

  sycl::free(nodes, q);
  sycl::free(results, q);
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

BOOST_AUTO_TEST_CASE(two_queue) {
  sycl::queue q1{sycl::property_list{sycl::property::queue::in_order{}}};
  sycl::queue q2(q1.get_context(), q1.get_device(),
		 sycl::property_list{sycl::property::queue::in_order{}});
  std::size_t test_size = 128;
  int *dev_ptr1 = sycl::malloc_device<int>(test_size, q1);
  int *dev_ptr2 = sycl::malloc_device<int>(test_size, q1);

  q1.memset(dev_ptr1, 0, test_size * sizeof(int));
  q2.memset(dev_ptr2, 0, test_size * sizeof(int));

  const unsigned iterations = 32;
  for (unsigned i = 0; i < iterations; i++) {
    q1.parallel_for(sycl::range<1>{test_size},
                    [=](sycl::id<1> idx) { dev_ptr1[idx] += 1; });

    q2.parallel_for(sycl::range<1>{test_size},
                    [=](sycl::id<1> idx) { dev_ptr2[idx] += 1; });
  }
  q1.wait();
  q2.wait();

  std::vector<int> host_ptr1(test_size);
  std::vector<int> host_ptr2(test_size);
  q1.memcpy(host_ptr1.data(), dev_ptr1, test_size * sizeof(int)).wait();
  q1.memcpy(host_ptr2.data(), dev_ptr2, test_size * sizeof(int));
  q1.wait();

  for (int i = 0; i < test_size; ++i) {
    BOOST_TEST(host_ptr1[i] == iterations);
    BOOST_TEST(host_ptr2[i] == iterations);
  }

  sycl::free(dev_ptr1, q1);
  sycl::free(dev_ptr2, q2);
}

=======
>>>>>>> 69cc2cf4 (Add LIT SSCP testing)
BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
