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

#include "../sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(smoke_task_queue_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(queue_fill) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  const int fill_value = 42;
  std::vector<int> host_buff(size);
  {
    sycl::buffer<int> buff{size};
    buff.set_final_data(host_buff.data());
    q.submit([&](sycl::handler &cgh) {
      auto buff_acc = buff.template get_access<sycl::access::mode::write>(cgh);
      cgh.fill(buff_acc, fill_value);
    });
    q.wait();
  }

  for (int i = 0; i < host_buff.size(); ++i) {
    BOOST_CHECK(host_buff[i] == fill_value);
  }
}

BOOST_AUTO_TEST_CASE(queue_copy) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  {
    sycl::buffer<int> in_buff{in_data.data(), size};
    sycl::buffer<int> out_buff{size};
    out_buff.set_final_data(out_data.data());
    q.submit([&](sycl::handler &cgh) {
      auto in_acc = in_buff.template get_access<sycl::access::mode::read>(cgh);
      auto out_acc =
          out_buff.template get_access<sycl::access::mode::write>(cgh);
      cgh.copy(in_acc, out_acc);
    });
    q.wait();
  }

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(in_data[i] == out_data[i]);
  }
}

BOOST_AUTO_TEST_CASE(queue_copy_offset) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> in_data(size * 2);
  std::vector<int> out_data(size * 2);
  for (int i = 0; i < size * 2; ++i) {
    in_data[i] = i;
    out_data[i] = 0;
  }

  {
    sycl::buffer<int> in_buff{in_data.data(), size * 2};
    sycl::buffer<int> out_buff{out_data.data(), size * 2};
    q.submit([&](sycl::handler &cgh) {
      auto in_acc = in_buff.template get_access<sycl::access::mode::read>(
          cgh, sycl::range<1>(size), sycl::id<1>(size));
      auto out_acc = out_buff.template get_access<sycl::access::mode::write>(
          cgh, sycl::range<1>(size), sycl::id<1>(size));

      cgh.copy(in_acc, out_acc);
    });
    q.wait();
  }

  for (int i = 0; i < 2 * size; ++i) {
    if (i < size) {
      BOOST_CHECK(0 == out_data[i]);
    } else {
      BOOST_CHECK(in_data[i] == out_data[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(queue_const_acc) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  {
    sycl::buffer<int> a_buff{in_data.data(), size};
    sycl::buffer<int> c_buff{size};
    c_buff.set_final_data(out_data.data());

    sycl::nd_range<1> b{sycl::range<1>(size), sycl::range<1>(4)};
    q.submit([&](sycl::handler &cgh) {
      auto const_acc =
          a_buff.template get_access<sycl::access_mode::read,
                                     sycl::target::constant_buffer>(cgh);

      auto out_acc = c_buff.template get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(size), sycl::range<1>(128)),
          [=](sycl::nd_item<1> id) {
            auto idx = id.get_global_id(0);
            out_acc[idx] = const_acc[idx];
          });
    });
    q.wait();
  }

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(in_data[i] == out_data[i]);
    if (in_data[i] != out_data[i])
      break;
  }
}

BOOST_AUTO_TEST_CASE(queue_kernel) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  {
    sycl::buffer<int> a_buff{in_data.data(), size};
    sycl::buffer<int> b_buff{size};
    sycl::buffer<int> c_buff{size};
    c_buff.set_final_data(out_data.data());

    sycl::nd_range<1> a{sycl::range<1>(size), sycl::range<1>(128)};
    sycl::nd_range<1> b{sycl::range<1>(size), sycl::range<1>(4)};
    for (unsigned int i = 0; i < 2; i++) {
      q.submit([&](sycl::handler &cgh) {
        auto in_acc =
            i == 0 ? a_buff.template get_access<sycl::access::mode::read>(cgh)
                   : b_buff.template get_access<sycl::access::mode::read>(cgh);

        auto out_acc =
            i == 0 ? b_buff.template get_access<sycl::access::mode::write>(cgh)
                   : c_buff.template get_access<sycl::access::mode::write>(cgh);

        sycl::nd_range<1> r = i == 0 ? a : b;
        cgh.parallel_for(r, [=](sycl::nd_item<1> id) {
          auto idx = id.get_global_id(0);
          out_acc[idx] = in_acc[idx];
        });
      });
    }
    q.wait();
  }

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(in_data[i] == out_data[i]);
    if (in_data[i] != out_data[i])
      break;
  }
}

BOOST_AUTO_TEST_CASE(queue_prefetch) {
  sycl::queue q{sycl::property::queue::in_order{}};
  size_t size = 1024;

  int *ptr;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    ptr = sycl::malloc_shared<int>(size, q);
  } else {
    ptr = sycl::malloc_device<int>(size, q);
  }

  q.prefetch(ptr, size * sizeof(int)).wait();
  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_memset) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;

  int *ptrA = sycl::malloc_device<int>(size, q);
  int *ptrB = sycl::malloc_device<int>(size, q);

  int pattern = 42;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> id) { ptrA[id] = pattern; });
  });

  q.memset(ptrB, 0, size * sizeof(int));

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> id) { ptrB[id] = ptrA[id]; });
  });

  q.memset(ptrA, 0, size * sizeof(int));

  std::vector<int> A_data(size);
  std::vector<int> B_data(size);
  q.memcpy(A_data.data(), ptrA, size * sizeof(int));
  q.memcpy(B_data.data(), ptrB, size * sizeof(int));
  q.wait();

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(0 == A_data[i]);
    BOOST_CHECK(pattern == B_data[i]);
    if (0 != A_data[i] || pattern != B_data[i])
      break;
  }
  sycl::free(ptrA, q);
  sycl::free(ptrB, q);
}

BOOST_AUTO_TEST_CASE(queue_usm) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> out_data(size);

  int pattern = 42;
  int *ptr = sycl::malloc_device<int>(size, q);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> id) { ptr[id] = pattern; });
  });
  q.memcpy(out_data.data(), ptr, size * sizeof(int));
  q.wait();

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(pattern == out_data[i]);
    if (pattern != out_data[i])
      break;
  }
  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_usm_offset) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 1024;
  std::vector<int> in_data(2 * size);
  std::vector<int> out_data(size);
  for (unsigned i = 0; i < 2 * size; i++) {
    in_data[i] = i;
  }

  int *src_ptr = sycl::malloc_device<int>(size * 2, q);
  int *src_offset_ptr = src_ptr + size;

  int *dst_ptr = sycl::malloc_device<int>(size * 2, q);
  int *dst_offset_ptr = dst_ptr + size;
  q.memcpy(src_offset_ptr, in_data.data() + size, size * sizeof(int));
  q.memcpy(dst_offset_ptr, src_offset_ptr, size * sizeof(int));
  q.memcpy(out_data.data(), dst_offset_ptr, size * sizeof(int));
  q.wait();

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(size + i == out_data[i]);
    if (size + i != out_data[i])
      break;
  }
  sycl::free(src_ptr, q);
  sycl::free(dst_ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_struct) {
  sycl::queue q{sycl::property::queue::in_order{}};

  constexpr unsigned size = 256;
  constexpr unsigned arr_size = size / 2;
  std::vector<int> out_data(size);

  struct foo {
    int A[arr_size];
  } bar;

  for (unsigned i = 0; i < arr_size; i++) {
    bar.A[i] = i;
  }

  int *ptr = sycl::malloc_device<int>(size, q);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> id) { ptr[id] = bar.A[id / 2]; });
  });
  q.memcpy(out_data.data(), ptr, size * sizeof(int));
  q.wait();

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(i / 2 == out_data[i]);
    if (i / 2 != out_data[i])
      break;
  }
  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_local) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 512;
  int local_size = 128;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  {
    sycl::buffer<int> a_buff{in_data.data(), size};
    sycl::buffer<int> c_buff{size};
    c_buff.set_final_data(out_data.data());

    q.submit([&](sycl::handler &cgh) {
      auto in_acc = a_buff.template get_access<sycl::access::mode::read>(cgh);
      auto out_acc = c_buff.template get_access<sycl::access::mode::write>(cgh);

      auto scratch = sycl::local_accessor<int, 1>{local_size, cgh};

      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(size), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> id) {
            auto gid = id.get_global_id(0);
            auto lid = id.get_local_id(0);

            scratch[lid] = in_acc[gid];
            id.barrier();
            if (lid == 0) {
              out_acc[gid] = 0;
            } else {
              out_acc[gid] = scratch[lid - 1];
            }
          });
    });
    q.wait();
  }

  for (int i = 0; i < size; ++i) {
    int lid = i % local_size;
    int group = i / local_size;
    int res = lid == 0 ? 0 : (i - 1);
    BOOST_CHECK(res == out_data[i]);
    if (res != out_data[i])
      break;
  }
}

BOOST_AUTO_TEST_CASE(queue_offset_subtract) {
  sycl::queue q{sycl::property::queue::in_order{}};

  // See Issue 7 in doc/vulkan.md
  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Skipping due to issue with physical addressing");
    return;
  }

  size_t size = 128;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  int offset = 4;
  int *in_ptr = sycl::malloc_device<int>(size, q);
  int *out_ptr = sycl::malloc_device<int>(size, q);

  q.memcpy(in_ptr, in_data.data(), size * sizeof(int));
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<1>(size),
        [=](sycl::id<1> id) {
          out_ptr[id] = in_ptr[offset - 1];
        });
  });
  q.memcpy(out_data.data(), out_ptr, size * sizeof(int));
  q.wait();

  const int result = offset - 1;
  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(result == out_data[i]);
    if (result != out_data[i])
      break;
  }

  sycl::free(in_ptr, q);
  sycl::free(out_ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_private_address_store_to_global) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
    return;
  }

  // We just check the kernel compiles, there is no invalid
  // value that `*ptr` could contain.
  int **ptr = sycl::malloc_device<int *>(1, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       int arr;
       *ptr = &arr;
     });
   }).wait();

  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_local_address_store_to_global) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
    return;
  }

  // We just check the kernel compiles, there is no invalid
  // value that `*ptr` could contain.
  int **ptr = sycl::malloc_device<int *>(1, q);
  q.submit([&](sycl::handler &cgh) {
     auto scratch = sycl::local_accessor<int, 1>{1, cgh};
     cgh.single_task([=]() { *ptr = &scratch[0]; });
   }).wait();

  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_global_address_store_to_global) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
    return;
  }

  int **ptr = sycl::malloc_device<int *>(1, q);
  int *target_ptr = sycl::malloc_device<int>(1, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() { *ptr = target_ptr; });
  });

  int *host;
  q.memcpy(&host, ptr, sizeof(int *)).wait();

  BOOST_CHECK(target_ptr == host);

  sycl::free(ptr, q);
  sycl::free(target_ptr, q);
}

BOOST_AUTO_TEST_CASE(queue_linked_list) {
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().get_backend() == sycl::backend::vk) {
    BOOST_TEST_MESSAGE("Vulkan doesn't support pointer to pointer args");
    return;
  }

  struct Node {
    int id;
    Node *ptr;
  };

  Node *nodes = sycl::malloc_device<Node>(3, q);
  Node nodeC{2, nullptr};
  Node nodeB{1, nodes + 2};
  Node nodeA{0, nodes + 1};

  q.memcpy(nodes, &nodeA, sizeof(Node));
  q.memcpy(nodes + 1, &nodeB, sizeof(Node));
  q.memcpy(nodes + 2, &nodeC, sizeof(Node));
  q.wait();

  int *ptr = sycl::malloc_device<int>(1, q);
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      Node *curr = nodes;
      while (curr != nullptr) {
        *ptr = curr->id;
        curr = curr->ptr;
      }
    });
  });

  int result;
  q.memcpy(&result, ptr, sizeof(int)).wait();

  BOOST_CHECK(2 == result);

  sycl::free(ptr, q);
  sycl::free(nodes, q);
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
