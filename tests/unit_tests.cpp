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

#include <tuple>

#define BOOST_MPL_CFG_GPU_ENABLED // Required for nvcc
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE hipsycl unit tests
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list_c.hpp>
#include <boost/mpl/list.hpp>

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

struct reset_device_fixture {
  ~reset_device_fixture() {
    cl::sycl::detail::application::reset();
  }
};

BOOST_FIXTURE_TEST_SUITE(device_test_suite, reset_device_fixture)

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

BOOST_AUTO_TEST_CASE(basic_parallel_for_with_offset) {
  constexpr size_t num_threads = 128;
  constexpr size_t offset = 64;
  cl::sycl::queue queue;
  std::vector<int> host_buf(num_threads + offset, 0);
  cl::sycl::buffer<int, 1> buf{host_buf.data(),
    cl::sycl::range<1>(num_threads + offset)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class basic_parallel_for_with_offset>(
      cl::sycl::range<1>(num_threads),
      cl::sycl::id<1>(offset),
      [=](cl::sycl::item<1> tid) {
        acc[tid] = tid[0];
      });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads + offset; ++i) {
    BOOST_REQUIRE(acc[i] == (i >= offset ? i : 0));
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

BOOST_AUTO_TEST_CASE(buffer_versioning) {
  namespace s = cl::sycl;
  constexpr size_t buf_size = 32;

  s::queue queue;
  s::buffer<int, 1> buf(buf_size);
  {
    auto acc = buf.get_access<s::access::mode::discard_write>();
    for(int i = 0; i < buf_size; ++i) {
      acc[i] = i;
    }
  }

  queue.submit([&](s::handler& cgh) {
    auto acc = buf.get_access<s::access::mode::discard_write>(cgh);
    cgh.parallel_for<class buffer_versioning>(buf.get_range(), [=](s::id<1> id) {
      acc[id] = buf_size - id[0];
    });
  });

  {
    auto acc = buf.get_access<s::access::mode::read>();
    for(int i = 0; i < buf_size; ++i) {
      BOOST_REQUIRE(acc[i] == buf_size - i);
    }
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

using test_dimensions = boost::mpl::list_c<int, 1, 2, 3>;

template<int dimensions, template<int D> class T>
void assert_array_equality(const T<dimensions>& a, const T<dimensions>& b) {
  if(dimensions >= 1) BOOST_REQUIRE(a[0] == b[0]);
  if(dimensions >= 2) BOOST_REQUIRE(a[1] == b[1]);
  if(dimensions == 3) BOOST_REQUIRE(a[2] == b[2]);
}

template <template<int D> class T, int dimensions>
auto make_test_value(const T<1>& a, const T<2>& b, const T<3>& c) {
  return std::get<dimensions - 1>(std::make_tuple(a, b, c));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range_api, _dimensions, test_dimensions::type) {
  namespace s = cl::sycl;
  constexpr auto d = _dimensions::value;

  const auto test_value = make_test_value<s::range, d>({ 5 }, { 5, 7 }, { 5, 7, 11 });

  // --- Common by-value semantics ---

  {
    // Copy constructor
    s::range<d> range(test_value);
    assert_array_equality(range, test_value);
  }
  {
    // Move constructor
    s::range<d> range(([&]() {
      s::range<d> copy(test_value);
      return std::move(copy);
    })());
    assert_array_equality(range, test_value);
  }
  {
    // Copy assignment
    s::range<d> range;
    range = test_value;
    assert_array_equality(range, test_value);
  }
  {
    // Move assignment
    s::range<d> range;
    range = ([&]() {
      s::range<d> copy(test_value);
      return std::move(copy);
    })();
    assert_array_equality(range, test_value);
  }
  {
    // Equality operator
    s::range<d> range;
    BOOST_TEST(!(range == test_value));
    range = test_value;
    BOOST_TEST((range == test_value));
  }
  {
    // Inequality operator
    s::range<d> range;
    BOOST_TEST((range != test_value));
    range = test_value;
    BOOST_TEST(!(range != test_value));
  }

  // --- range-specific API ---

  {
    // range::get()
    const auto range = test_value;
    if(d >= 1) BOOST_TEST(range.get(0) == 5);
    if(d >= 2) BOOST_TEST(range.get(1) == 7);
    if(d == 3) BOOST_TEST(range.get(2) == 11);
  }
  {
    // range::operator[]
    auto range = test_value;
    if(d >= 1) range[0] += 2;
    if(d >= 2) range[1] += 3;
    if(d == 3) range[2] += 5;
    assert_array_equality(range, make_test_value<s::range, d>(
      { 7 }, { 7, 10 }, { 7, 10, 16 }));
  }
  {
    // const range::operator[]
    const auto range = test_value;
    if(d >= 1) BOOST_TEST(range[0] == 5);
    if(d >= 2) BOOST_TEST(range[1] == 7);
    if(d == 3) BOOST_TEST(range[2] == 11);
  }
  {
    // range::size()
    const auto range = test_value;
    BOOST_TEST(range.size() == 5 * (d >= 2 ? 7 : 1) * (d == 3 ? 11 : 1));
  }

  // TODO: In-place and binary operators
}

BOOST_AUTO_TEST_CASE_TEMPLATE(id_api, _dimensions, test_dimensions::type) {
  namespace s = cl::sycl;
  constexpr auto d = _dimensions::value;

  const auto test_value = make_test_value<s::id, d>(
    { 5 }, { 5, 7 }, { 5, 7, 11 });

  // --- Common by-value semantics ---

  {
    // Copy constructor
    s::id<d> id(test_value);
    assert_array_equality(id, test_value);
  }
  {
    // Move constructor
    s::id<d> id(([&]() {
      s::id<d> copy(test_value);
      return std::move(copy);
    })());
    assert_array_equality(id, test_value);
  }
  {
    // Copy assignment
    s::id<d> id;
    id = test_value;
    assert_array_equality(id, test_value);
  }
  {
    // Move assignment
    s::id<d> id;
    id = ([&]() {
      s::id<d> copy(test_value);
      return std::move(copy);
    })();
    assert_array_equality(id, test_value);
  }
  {
    // Equality operator
    s::id<d> id;
    BOOST_TEST(!(id == test_value));
    id = test_value;
    BOOST_TEST((id == test_value));
  }
  {
    // Inequality operator
    s::id<d> id;
    BOOST_TEST((id != test_value));
    id = test_value;
    BOOST_TEST(!(id != test_value));
  }

  // --- id-specific API ---

  {
    const auto test_range = make_test_value<s::range, d>(
      { 5 }, { 5, 7 }, { 5, 7, 11 });
    s::id<d> id{test_range};
    assert_array_equality(id, test_value);
  }
  {
    // TODO: Test conversion from item
    // (This is a bit annoying as items can only be constructed on a __device__)
  }
  {
    // id::get()
    const auto id = test_value;
    if(d >= 1) BOOST_TEST(id.get(0) == 5);
    if(d >= 2) BOOST_TEST(id.get(1) == 7);
    if(d == 3) BOOST_TEST(id.get(2) == 11);
  }
  {
    // id::operator[]
    auto id = test_value;
    if(d >= 1) id[0] += 2;
    if(d >= 2) id[1] += 3;
    if(d == 3) id[2] += 5;
    assert_array_equality(id, make_test_value<s::id, d>(
      { 7 }, { 7, 10 }, { 7, 10, 16 }));
  }
  {
    // const id::operator[]
    const auto id = test_value;
    if(d >= 1) BOOST_TEST(id[0] == 5);
    if(d >= 2) BOOST_TEST(id[1] == 7);
    if(d == 3) BOOST_TEST(id[2] == 11);
  }

  // TODO: In-place and binary operators
}

// Helper type to construct unique kernel names for all instantiations of
// a templated test case.
template<typename T, int dimensions, typename extra=T>
struct kernel_name {};

BOOST_AUTO_TEST_CASE_TEMPLATE(item_api, _dimensions, test_dimensions::type) {
  namespace s = cl::sycl;
  constexpr auto d = _dimensions::value;

  const auto test_range = make_test_value<s::range, d>(
    { 5 }, { 5, 7 }, { 5, 7, 11 });

  // TODO: Add tests for common by-value semantics

  s::queue queue;

  {
    // item::get_id and item::operator[] without offset

    s::buffer<s::id<d>, d> result1{test_range};
    s::buffer<s::id<d>, d> result2{test_range};
    s::buffer<s::id<d>, d> result3{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      auto acc3 = result3.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_id, d>>(test_range,
        [=](const s::item<d> item) {
          {
            acc1[item] = item.get_id();
          }
          {
            auto id2 = item.get_id();
            if(d >= 1) id2[0] = item.get_id(0);
            if(d >= 2) id2[1] = item.get_id(1);
            if(d == 3) id2[2] = item.get_id(2);
            acc2[item] = id2;
          }
          {
            auto id3 = item.get_id();
            if(d >= 1) id3[0] = item.get_id(0);
            if(d >= 2) id3[1] = item.get_id(1);
            if(d == 3) id3[3] = item.get_id(2);
            acc3[item] = id3;
          }
        });
      });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    auto acc3 = result3.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], id);
          assert_array_equality(acc2[id], id);
          assert_array_equality(acc3[id], id);
        }
      }
    }
  }
  {
    // item::get_id and item::operator[] with offset

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<s::id<d>, d> result1{test_range + test_offset};
    s::buffer<s::id<d>, d> result2{test_range + test_offset};
    s::buffer<s::id<d>, d> result3{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      auto acc3 = result3.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_id_offset, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          {
            acc1[item] = item.get_id();
          }
          {
            auto id2 = item.get_id();
            if(d >= 1) id2[0] = item.get_id(0);
            if(d >= 2) id2[1] = item.get_id(1);
            if(d == 3) id2[2] = item.get_id(2);
            acc2[item] = id2;
          }
          {
            auto id3 = item.get_id();
            if(d >= 1) id3[0] = item.get_id(0);
            if(d >= 2) id3[1] = item.get_id(1);
            if(d == 3) id3[3] = item.get_id(2);
            acc3[item] = id3;
          }
        });
    });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    auto acc3 = result3.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], id);
          assert_array_equality(acc2[id], id);
          assert_array_equality(acc3[id], id);
        }
      }
    }
  }
  {
    // item::get_range

    s::buffer<s::range<d>, d> result1{test_range};
    s::buffer<s::range<d>, d> result2{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_range, d>>(test_range,
        [=](const s::item<d> item) {
          {
            acc1[item] = item.get_range();
          }
          {
            auto range2 = item.get_range();
            if(d >= 1) range2[0] = item.get_range(0);
            if(d >= 2) range2[1] = item.get_range(1);
            if(d == 3) range2[2] = item.get_range(2);
            acc2[item] = range2;
          }
        });
      });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], test_range);
          assert_array_equality(acc2[id], test_range);
        }
      }
    }
  }
  {
    // item::get_offset

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<s::id<d>, d> result{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_offset, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          acc[item] = item.get_offset();
        });
    });

    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc[id], s::id<d>(test_offset));
        }
      }
    }
  }
  {
    // Conversion operator from item<d, false> to item<d, true>

    s::buffer<s::id<d>, d> result{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_conversion, d>>(test_range,
        [=](const s::item<d> item) {
          acc[item] = item.get_offset();
        });
    });

    const auto empty_offset = make_test_value<s::id, d>({0}, {0, 0}, {0, 0, 0});
    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc[id], empty_offset);
        }
      }
    }
  }
  {
    // item::get_linear_id

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<size_t, d> result{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_linear_id, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          acc[item] = item.get_linear_id();
        });
    });

    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          const size_t linear_id = i * (jb - ja) * (kb - ka) + j * (kb - ka) + k;
          BOOST_REQUIRE(acc[id] == linear_id);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(explicit_buffer_copy_host_ptr, _dimensions,
  test_dimensions::type) {
  namespace s = cl::sycl;
  constexpr auto d = _dimensions::value;

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
  _dimensions, test_dimensions::type) {
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
  _dimensions, test_dimensions::type) {
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
  _dimensions, test_dimensions::type) {
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
  _dimensions, test_dimensions::type) {
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

/// Math function tests ///////////////////////////////////////////////////////

// list of types classified as "genfloat" in the SYCL standard
using math_test_genfloats = boost::mpl::list<
  float,
  cl::sycl::vec<float, 1>,
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<float, 8>,
  cl::sycl::vec<float, 16>,
  double,
  cl::sycl::vec<double, 1>,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>,
  cl::sycl::vec<double, 8>,
  cl::sycl::vec<double, 16>>;

namespace {

  template<typename DT, int N>
  using vec = cl::sycl::vec<DT, N>;
  auto tolerance = boost::test_tools::tolerance(0.0001);

  // utility type traits for generic testing

  template<typename T>
  struct vector_length {
    static constexpr int value = 0;
  };
  template<typename DT, int N>
  struct vector_length<vec<DT, N>> {
    static constexpr int value = N;
  };
  template<typename T>
  constexpr int vector_length_v = vector_length<T>::value;

  template<typename T>
  struct vector_dim {
    static constexpr int value = 0;
  };
  template<typename DT, int N>
  struct vector_dim<vec<DT, N>> {
    static constexpr int value = 1;
  };
  template<typename T>
  constexpr int vector_dim_v = vector_dim<T>::value;

  template<typename T>
  struct vector_elem {
    using type = T;
  };
  template<typename DT, int D>
  struct vector_elem<vec<DT, D>> {
    using type = DT;
  };
  template<typename T>
  using vector_elem_t = typename vector_elem<T>::type;

  template<typename TARGET_DT, typename T>
  struct vector_coerce_elem {
    using type = TARGET_DT;
  };
  template<typename TARGET_DT, typename DT, int D>
  struct vector_coerce_elem<TARGET_DT, vec<DT, D>> {
    using type = vec<TARGET_DT, D>;
  };
  template<typename TARGET_DT, typename T>
  using vector_coerce_elem_t = typename vector_coerce_elem<TARGET_DT, T>::type;

  // utility functions for generic testing

  template<typename DT, int D, std::enable_if_t<D<=4, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return std::get<D>(std::make_tuple(
      v.s0(),
      vec<DT, 1>(v.s0()),
      vec<DT, 2>(v.s0(), v.s1()),
      vec<DT, 3>(v.s0(), v.s1(), v.s2()),
      vec<DT, 4>(v.s0(), v.s1(), v.s2(), v.s3())));
  }
  template<typename DT, int D, std::enable_if_t<D==8, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return vec<DT, 8>(v.s0(), v.s1(), v.s2(), v.s3(), v.s4(), v.s5(), v.s6(), v.s7());
  }
  template<typename DT, int D, std::enable_if_t<D==16, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return v;
  }

  // runtime indexed access to vector elements
  // this could be a single function with constexpr if in C++17
  // could also be done by using knowledge of the internal structure,
  // but I wanted to avoid that.
  template<typename T, std::enable_if_t<vector_length_v<T> == 0, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx == 0);
    return v;
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 1, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    return v.x();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 2, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    return v.y();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 3, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    if(idx==1) return v.y();
    return v.z();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 4, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    if(idx==1) return v.y();
    if(idx==2) return v.z();
    return v.w();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 8, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.s0();
    if(idx==1) return v.s1();
    if(idx==2) return v.s2();
    if(idx==3) return v.s3();
    if(idx==4) return v.s4();
    if(idx==5) return v.s5();
    if(idx==6) return v.s6();
    if(idx==7) return v.s7();
    return v.s7();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 16, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.s0();
    if(idx==1) return v.s1();
    if(idx==2) return v.s2();
    if(idx==3) return v.s3();
    if(idx==4) return v.s4();
    if(idx==5) return v.s5();
    if(idx==6) return v.s6();
    if(idx==7) return v.s7();
    if(idx==8) return v.s8();
    if(idx==9) return v.s9();
    if(idx==10) return v.sA();
    if(idx==11) return v.sB();
    if(idx==12) return v.sC();
    if(idx==13) return v.sD();
    if(idx==14) return v.sE();
    if(idx==15) return v.sF();
    return v.sF();
  }

  // reference functions

  double ref_clamp(double a, double min, double max) { // in C++17 <algorithm>, remove on upgrade
    if(a < min) return min;
    if(a > max) return max;
    return a;
  }

  static constexpr double pi = 3.1415926535897932385;
  double ref_degrees(double v) {
    return 180.0/pi * v;
  }
  double ref_radians(double v) {
    return pi/180.0 * v;
  }

  double ref_mix(double x, double y, double a) {
    return x + (y - x) * a;
  }

  double ref_step(double edge, double x) {
    if(x < edge) return 0.0;
    return 1.0;
  }

  double ref_smoothstep(double edge0, double edge1, double x) {
    BOOST_REQUIRE(edge0 < edge1); // Standard: results are undefined if edge0 >= edge1
    if(x <= edge0) return 0.0;
    if(x >= edge1) return 1.0;
    double t = ref_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double ref_sign(double x) {
    if(x > 0.0) return 1.0;
    if(x < 0.0) return -1.0;
    if(std::isnan(x)) return 0.0;
    return x;
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_binary, T,
                              math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 8;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class math_binary, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::atan2(acc[0], acc[1]);
      acc[i++] = s::copysign(acc[0], acc[1]);
      acc[i++] = s::fdim(acc[0], acc[1]);
      acc[i++] = s::fmin(acc[0], acc[1]);
      acc[i++] = s::fmax(acc[0], acc[1]);
      acc[i++] = s::fmod(acc[0], acc[1]);
      acc[i++] = s::hypot(acc[0], acc[1]);
      acc[i++] = s::pow(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = vector_dim_v<T>; c <= D; ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == atan2(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == copysign(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == fdim(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == fmin(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == fmax(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == fmod(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == hypot(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == pow(comp(acc[0], c), comp(acc[1], c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(common_functions, T,
    math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 20;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  constexpr DT input_scalar = 3.5f;
  constexpr DT mix_input_1 = 0.5f;
  constexpr DT mix_input_2 = 0.8f;
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions
  // some of these are tested multiple times to ensure that all overloads are covered
  // (e.g. combinations of vec and scalar input)

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class common_functions, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::clamp(acc[0], acc[1], acc[1] + static_cast<DT>(10));
      acc[i++] = s::clamp(acc[0], input_scalar, static_cast<DT>(input_scalar + 10));
      acc[i++] = s::degrees(acc[0]);
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], input_scalar);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::min(acc[0], input_scalar);
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_2});
      acc[i++] = s::mix(acc[0], acc[1], mix_input_1);
      acc[i++] = s::radians(acc[0]);
      acc[i++] = s::step(acc[0], acc[1]);
      acc[i++] = s::step(input_scalar, acc[0]);
      acc[i++] = s::smoothstep(acc[0], acc[0] + static_cast<DT>(10), acc[1]);
      acc[i++] = s::smoothstep(input_scalar, input_scalar + 1, acc[0]);
      acc[i++] = s::sign(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = vector_dim_v<T>; c <= D; ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), comp(acc[1], c), comp(acc[1], c) + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), input_scalar, input_scalar + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_degrees(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_2), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_radians(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(input_scalar, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(comp(acc[0], c), comp(acc[0], c) + 10, comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(input_scalar, input_scalar + 1, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_sign(comp(acc[0], c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line

