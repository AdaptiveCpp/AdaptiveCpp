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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE hipsycl unit tests
#include <boost/test/unit_test.hpp>

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

BOOST_AUTO_TEST_CASE(basic_single_task) {
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(1)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class basic_single_task>([=]() {
      acc[0] = 321;
    });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  BOOST_TEST(acc[0] == 321);
}

BOOST_AUTO_TEST_CASE(basic_parallel_for) {
  constexpr size_t num_threads = 128;
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(num_threads)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class basic_parallel_for>(cl::sycl::range<1>(num_threads),
      [=](cl::sycl::item<1> tid) {
        acc[tid] = tid[0];
      });
    });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads; ++i) {
    BOOST_REQUIRE(acc[i] == i);
  }
}

BOOST_AUTO_TEST_CASE(basic_parallel_for_nd) {
  constexpr size_t num_threads = 128;
  constexpr size_t group_size = 16;
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(num_threads)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cl::sycl::nd_range<1> kernel_range{cl::sycl::range<1>(num_threads),
      cl::sycl::range<1>(group_size)};
    cgh.parallel_for<class basic_parallel_for_nd>(kernel_range,
      [=](cl::sycl::nd_item<1> tid) {
        acc[tid.get_global()[0]] = tid.get_group(0);
      });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads; ++i) {
    BOOST_REQUIRE(acc[i] == i / group_size);
  }
}

BOOST_AUTO_TEST_CASE(hierarchical_dispatch) {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for_work_group<class hierarchical_dispatch_reduction>(
        cl::sycl::range<1>{global_size / local_size},
        cl::sycl::range<1>{local_size},
        [=](cl::sycl::group<1> wg) {
          int scratch[local_size];
          wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] = acc[item.get_global_id()];
          });
          for(size_t i = local_size/2; i > 0; i /= 2) {
            wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
              const size_t lid = item.get_local_id()[0];
              if(lid < i) scratch[lid] += scratch[lid + i];
            });
          }
          wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
            const size_t lid = item.get_local_id()[0];
            if(lid == 0) acc[item.get_global_id()] = scratch[0];
          });
        });
    });
  }

  for(size_t i = 0; i < global_size / local_size; ++i) {
    size_t expected = 0;
    for(size_t j = 0; j < local_size; ++j) expected += i * local_size + j;
    size_t computed = host_buf[i * local_size];
    BOOST_TEST(computed == expected);
  }
}

BOOST_AUTO_TEST_CASE(dynamic_local_memory) {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler& cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>
        {local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
        cl::sycl::nd_range<1>{global_size, local_size},
        [=](cl::sycl::nd_item<1> item) {
          const auto lid = item.get_local(0);
          scratch[lid] = acc[item.get_global()];
          item.barrier();
          for(size_t i = local_size/2; i > 0; i /= 2) {
            if(lid < i) scratch[lid] += scratch[lid + i];
            item.barrier();
          }
          if(lid == 0) acc[item.get_global()] = scratch[0];
        });
    });
  }

  for(size_t i = 0; i < global_size / local_size; ++i) {
    size_t expected = 0;
    for(size_t j = 0; j < local_size; ++j) expected += i * local_size + j;
    size_t computed = host_buf[i * local_size];
    BOOST_TEST(computed == expected);
  }
}

BOOST_AUTO_TEST_CASE(placeholder_accessors) {
  using namespace cl::sycl::access;
  constexpr size_t num_elements = 4096 * 1024;

  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{num_elements};

  {
    auto acc = buf.get_access<mode::discard_write>();
    for(size_t i = 0; i < num_elements; ++i) acc[i] = static_cast<int>(i);
  }

  cl::sycl::accessor<int, 1, mode::read_write, target::global_buffer, placeholder::true_t>
    ph_acc{buf};

  queue.submit([&](cl::sycl::handler& cgh) {
    auto ph_acc_copy = ph_acc; // Test that placeholder accessors can be copied
    cgh.require(ph_acc_copy);
    cgh.parallel_for<class placeholder_accessors>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        ph_acc_copy[tid] *= 2;
      });
  });

  {
    auto acc = buf.get_access<mode::read>();
    for(size_t i = 0; i < num_elements; ++i) {
      BOOST_REQUIRE(acc[i] == 2 * i);
    }
  }
}

BOOST_AUTO_TEST_CASE(task_graph_synchronization) {
  using namespace cl::sycl::access;
  constexpr size_t num_elements = 4096 * 1024;

  cl::sycl::queue q1;
  cl::sycl::queue q2;
  cl::sycl::queue q3;

  cl::sycl::buffer<int, 1> buf_a{num_elements};
  cl::sycl::buffer<int, 1> buf_b{num_elements};
  cl::sycl::buffer<int, 1> buf_c{num_elements};

  q1.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_init_a>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_a[tid] = static_cast<int>(tid.get(0));
      });
  });

  q2.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::read>(cgh);
    auto acc_b = buf_b.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_init_b>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_b[tid] = acc_a[tid];
      });
  });

  q3.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::read>(cgh);
    auto acc_b = buf_b.get_access<mode::read>(cgh);
    auto acc_c = buf_c.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_add_a_b>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_c[tid] = acc_a[tid] + acc_b[tid];
      });
  });

  auto result = buf_c.get_access<mode::read>();
  for(size_t i = num_elements; i < num_elements; ++i) {
    BOOST_REQUIRE(result[i] == 2 * i);
  }
}

BOOST_AUTO_TEST_CASE(vec_api) {
  cl::sycl::queue queue;
  cl::sycl::buffer<float, 1> results{60};

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = results.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task([=]() {
      size_t offset = 0;
      const auto store_results = [=, &offset](
        const std::initializer_list<float>& results) {
          for(auto r : results) {
            acc[offset++] = r;
          }
        };

      const cl::sycl::vec<float, 4> v1(1.0f);
      store_results({v1.x(), v1.y(), v1.z(), v1.w()});

      const cl::sycl::vec<float, 8> v2(1.f, 2.f, 4.f, v1, 8.f);
      store_results({v2.s0(), v2.s1(), v2.s2(), v2.s3(),
        v2.s4(), v2.s5(), v2.s6(), v2.s7()});

      // Broadcasting math functions over vector elements
      const auto v3 = cl::sycl::log2(v2);
      store_results({v3.s0(), v3.s1(), v3.s2(), v3.s3(),
        v3.s4(), v3.s5(), v3.s6(), v3.s7()});

      const auto v4 = cl::sycl::fma(v2, v2, v2);
      store_results({v4.s0(), v4.s1(), v4.s2(), v4.s3(),
        v4.s4(), v4.s5(), v4.s6(), v4.s7()});

      const auto v5 = v4 + v4;
      store_results({v5.s0(), v5.s1(), v5.s2(), v5.s3(),
        v5.s4(), v5.s5(), v5.s6(), v5.s7()});

      const auto v6 = cl::sycl::cos(v2) < cl::sycl::sin(v2);
      store_results({(float)v6.s0(), (float)v6.s1(), (float)v6.s2(), (float)v6.s3(),
        (float)v6.s4(), (float)v6.s5(), (float)v6.s6(), (float)v6.s7()});

      // Swizzles!
      const cl::sycl::vec<float, 4> v7 = v2.lo();
      store_results({v7.x(), v7.y(), v7.z(), v7.w()});

      const cl::sycl::vec<float, 4> v8 = v7.wzyx();
      store_results({v8.x(), v8.y(), v8.z(), v8.w()});

      cl::sycl::vec<float, 4> v9(1.f);
      v9.xwyz() = v7.xxyy();
      store_results({v9.x(), v9.y(), v9.z(), v9.w()});

      // Nested swizzles
      cl::sycl::vec<float, 4> v10(4.f);
      v10.zxyw().lo() = v7.xxyy().hi();
      store_results({v10.x(), v10.y(), v10.z(), v10.w()});
    });
  });

  auto acc = results.get_access<cl::sycl::access::mode::read>();
  size_t offset = 0;
  const auto verify_results = [&](const std::initializer_list<float>& expected) {
    for(auto e : expected) {
      BOOST_TEST(acc[offset++] == e);
    }
  };

  verify_results({1.f, 1.f, 1.f, 1.f});                         // v1
  verify_results({1.f, 2.f, 4.f, 1.f, 1.f, 1.f, 1.f, 8.f});     // v2
  verify_results({0.f, 1.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f});     // v3
  verify_results({2.f, 6.f, 20.f, 2.f, 2.f, 2.f, 2.f, 72.f});   // v4
  verify_results({4.f, 12.f, 40.f, 4.f, 4.f, 4.f, 4.f, 144.f}); // v5
  verify_results({1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f});     // v6
  verify_results({1.f, 2.f, 4.f, 1.f});                         // v7
  verify_results({1.f, 4.f, 2.f, 1.f});                         // v8
  verify_results({1.f, 2.f, 2.f, 1.f});                         // v9
  verify_results({2.f, 4.f, 2.f, 4.f});                         // v10
}

