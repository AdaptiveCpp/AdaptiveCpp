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


#include "hipSYCL/sycl/access.hpp"
#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

BOOST_FIXTURE_TEST_SUITE(buffer_tests, reset_device_fixture)


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

// TODO: Extend this
BOOST_AUTO_TEST_CASE(buffer_api) {
  namespace s = cl::sycl;

  s::buffer<std::int32_t, 1> buf_a(32);
  s::buffer<int, 1> buf_b(32);
  {
    auto host_a = buf_a.get_host_access();
    auto host_b = buf_b.get_host_access();
    for(size_t i = 0; i < host_a.size(); ++i) {
      host_a[i] = i;
      host_b[i] = -1;
    }
  }

  auto buf_c = buf_a;

  BOOST_REQUIRE(buf_a == buf_a);
  BOOST_REQUIRE(buf_a != buf_b);
  BOOST_REQUIRE(buf_a == buf_c);

  // compile test, that reinterpreted buffer uses rebound allocator
  s::buffer<std::uint16_t, 1, s::buffer_allocator<std::uint16_t>> buf_e = buf_a.reinterpret<std::uint16_t>(s::range<1>{64});
  s::buffer<std::uint32_t, 1, s::buffer_allocator<std::uint32_t>> buf_f = buf_a.reinterpret<std::uint32_t>();

  auto host_e = buf_e.get_host_access();
  auto host_f = buf_f.get_host_access();
  auto host_a = buf_a.get_host_access();
  for(size_t i = 0; i < host_a.size(); ++i) {
    BOOST_CHECK_EQUAL(host_a[i], static_cast<int>(host_e[i * 2]));
    BOOST_CHECK_EQUAL(host_a[i], static_cast<int>(host_f[i]));
  }
}


// TODO: Extend this
BOOST_AUTO_TEST_CASE(buffer_api_2d) {
  namespace s = cl::sycl;

  s::buffer<std::int32_t, 2> buf_a(s::range<2>{4, 4});
  s::buffer<std::int32_t, 2> buf_b(s::range<2>{4, 4});
  {
    auto host_a = buf_a.get_host_access();
    auto host_b = buf_b.get_host_access();
    for(size_t i = 0; i < host_a.get_range()[0]; ++i) {
      for(size_t j = 0; j < host_a.get_range()[1]; ++j) {
        host_a[i][j] = i;
        host_b[i][j] = -1;
      }
    }
  }

  auto buf_c = buf_a;

  BOOST_REQUIRE(buf_a == buf_a);
  BOOST_REQUIRE(buf_a != buf_b);
  BOOST_REQUIRE(buf_a == buf_c);

  // compile test, that reinterpreted buffer uses rebound allocator
  s::buffer<std::uint16_t, 1, s::buffer_allocator<std::uint16_t>> buf_d
    = buf_a.reinterpret<std::uint16_t>(s::range<1>{32});
  s::buffer<std::uint32_t, 1, s::buffer_allocator<std::uint32_t>> buf_e
    = buf_a.reinterpret<std::uint32_t, 1>();
  s::buffer<std::uint16_t, 2, s::buffer_allocator<std::uint16_t>> buf_f
    = buf_a.reinterpret<std::uint16_t>(s::range<2>{4, 8});
  s::buffer<std::uint32_t, 2, s::buffer_allocator<std::uint32_t>> buf_g
    = buf_a.reinterpret<std::uint32_t, 2>();

  auto host_d = buf_d.get_host_access();
  auto host_e = buf_e.get_host_access();
  auto host_f = buf_f.get_host_access();
  auto host_g = buf_g.get_host_access();
  auto host_a = buf_a.get_host_access();
  for(size_t i = 0; i < host_a.get_range()[0]; ++i) {
    for(size_t j = 0; j < host_a.get_range()[1]; ++j) {
      BOOST_CHECK_EQUAL(host_a[i][j], 
        static_cast<int>(host_d[i * host_a.get_range()[1] * 2 + j * 2]));
      BOOST_CHECK_EQUAL(host_a[i][j], 
        static_cast<int>(host_e[i * host_a.get_range()[1] + j]));
      BOOST_CHECK_EQUAL(host_a[i][j], static_cast<int>(host_f[i][j * 2]));
      BOOST_CHECK_EQUAL(host_a[i][j], static_cast<int>(host_g[i][j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(buffer_update_host) {
  cl::sycl::queue q;
  std::vector<int> host_buf(4);
  cl::sycl::buffer<int> sycl_buf(host_buf.data(), host_buf.size());

  q.submit([&](cl::sycl::handler& cgh) {
    auto acc = sycl_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class update_host_test>(sycl_buf.get_range(), [=](cl::sycl::item<1> item){
      acc[item] += item.get_id()[0];
    });
  });

  q.submit([&](cl::sycl::handler& cgh) {
    cgh.update_host(sycl_buf.get_access<cl::sycl::access::mode::read>(cgh));
  }).wait();

  BOOST_CHECK(host_buf == (std::vector{0, 1, 2, 3}));
}

BOOST_AUTO_TEST_CASE(buffer_external_writeback) {
  cl::sycl::queue q;

  std::size_t size = 1024;
  std::vector<int> host_buff(size);
  {
    cl::sycl::buffer<int> buff{size};

    buff.set_final_data(host_buff.data());

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          buff.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

      cgh.parallel_for<class buffer_external_writeback_test>(
          cl::sycl::range{size}, [=](cl::sycl::id<1> idx) {
            acc[idx.get(0)] = static_cast<int>(idx.get(0));
          });
    });
  }

  for(int i = 0; i < host_buff.size(); ++i) {
    BOOST_CHECK(host_buff[i] == i);
  }
}

BOOST_AUTO_TEST_CASE(buffer_external_writeback_nullptr) {
  cl::sycl::queue q;

  std::size_t size = 1024;
  std::vector<int> host_buff(size);
  {
    cl::sycl::buffer<int> buff{size};

    buff.set_final_data(nullptr);

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          buff.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

      cgh.parallel_for<class buffer_external_writeback_test>(
          cl::sycl::range{size}, [=](cl::sycl::id<1> idx) {
            acc[idx.get(0)] = static_cast<int>(idx.get(0));
          });
    });
  }

  for(int i = 0; i < host_buff.size(); ++i) {
    BOOST_CHECK(host_buff[i] == 0);
  }
}

BOOST_AUTO_TEST_CASE(buffer_const_ptr) {
  namespace s = cl::sycl;

  int initial_val = 1;
  int test_val = 42;

  std::size_t size = 1024;
  std::vector<int> host_buff(size, initial_val);
  {
    const int* host_ptr = host_buff.data();
    s::buffer<int> buff{host_ptr, size};

    s::queue{}.submit([&](s::handler &cgh) {
      auto acc =
        buff.get_access<s::access::mode::write>(cgh);

      cgh.parallel_for(s::range{size}, [=](auto idx) {
        acc[idx] = test_val;
      });
    });
  }

  // Here, host_buff should still contain the original data, since we passed
  // a const ptr to the buffer constructor.
  for (auto val : host_buff)
    BOOST_CHECK(val == initial_val);

  // Now do the same thing, but set a valid final data address
  {
    const int* host_ptr = host_buff.data();
    s::buffer<int> buff{host_ptr, size};

    int* no_const_host_ptr = host_buff.data();
    buff.set_final_data(no_const_host_ptr);

    s::queue{}.submit([&](s::handler &cgh) {
      auto acc =
        buff.get_access<s::access::mode::write>(cgh);

      cgh.parallel_for(s::range{size}, [=](auto idx) {
        acc[idx] = test_val;
      });
    });
  }

  // Here, host_buff should now contain the new data
  for (auto val : host_buff)
    BOOST_CHECK(val == test_val);
}

BOOST_AUTO_TEST_CASE(buffer_const_T_constructor) {
  namespace s = cl::sycl;

  std::size_t size = 1024;
  std::vector<int> host_buff(size);

  const int* host_ptr = host_buff.data();
  s::buffer<const int> buff{host_ptr, size};

  int *host_ptr2 = host_buff.data();
  s::buffer<const int> buff2{host_ptr2, size};
}

BOOST_AUTO_TEST_CASE(buffer_container_constructor) {
  cl::sycl::queue q;

  const int testVal = 42;

  std::size_t size = 1024;
  std::vector<int> host_buff(size, 0);
  {
    cl::sycl::buffer<int> buff{host_buff};

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc =
        buff.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for(cl::sycl::range{size}, [=](auto idx) {
        acc[idx] = testVal;
      });
    });
  }

  for(int i = 0; i < host_buff.size(); ++i) {
    BOOST_CHECK(host_buff[i] == testVal);
  }
}

BOOST_AUTO_TEST_CASE(buffer_container_constructor_no_def_constr) {
  cl::sycl::queue q;

  struct A {
    A() = delete;
    A(int val) : val(val) {}
    int val;
  };

  A testVal = A{42};
  std::array<A, 2> data1 = {A{1}, A{2}};
  std::array<A, 2> data2 = {A{1}, A{2}};
  {
    cl::sycl::buffer<A> buff1{data1};
    cl::sycl::buffer<A> buff2{data2.begin(), data2.end()};
    buff2.set_final_data(data2.data());

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc1 =
        buff1.get_access<cl::sycl::access::mode::write>(cgh);
      auto acc2 =
        buff2.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for(cl::sycl::range{2}, [=](auto idx) {
        acc1[idx] = testVal;
        acc2[idx] = testVal;
      });
    });
  }

  for(int i = 0; i < data1.size(); ++i)
    BOOST_CHECK(data1[i].val == testVal.val);
  for(int i = 0; i < data2.size(); ++i)
    BOOST_CHECK(data2[i].val == testVal.val);
}

BOOST_AUTO_TEST_CASE(buffer_shared_ptr) {
  namespace s = cl::sycl;
  s::queue q{};

  const std::size_t size = 1024;
  int testVal = 42;

  // Constructor that takes non-empty shared_ptr
  {
    std::shared_ptr<int> hostptr{new int[size], std::default_delete<int[]>{}};

    {
      s::buffer<int> buf{hostptr, size};

      // buffer must copy the shared_ptr, so its ref count should be two now
      BOOST_CHECK(hostptr.use_count() == 2);

      q.submit([&](auto &cgh) {
        auto acc = buf.get_access<s::access::mode::write>(cgh);
        cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
      });
    }

    std::vector<int> hostdata(hostptr.get(), hostptr.get() + size);
    for (auto val : hostdata)
      BOOST_CHECK(val == testVal);
  }

  // Constructor that takes empty shared_ptr
  {
    std::shared_ptr<int> hostptr;
    s::buffer<int> buf{hostptr, size};
  }

  // Constructor that takes empty shared_ptr, but set final data pointer
  {
    std::vector<int> hostdata(size, -1);
    std::shared_ptr<int> hostptr;

    {
      s::buffer<int> buf{hostptr, size};
      buf.set_final_data(hostdata.data());

      q.submit([&](auto &cgh) {
        auto acc = buf.get_access<s::access::mode::write>(cgh);
        cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
      });
    }

    for (auto val : hostdata)
      BOOST_CHECK(val == testVal);
  }

  // Constructor that takes a unique ptr
  {
    std::unique_ptr<int, std::default_delete<int[]>> hostptr{
        new int[size], std::default_delete<int[]>{}};

    s::buffer<int> buf{std::move(hostptr), size};

    BOOST_CHECK(!hostptr);

    q.submit([&](auto &cgh) {
      auto acc = buf.get_access<s::access::mode::write>(cgh);
      cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
    });

    auto ha = buf.get_host_access();
    for (auto val : ha)
      BOOST_CHECK(val == testVal);
  }

  // Set final data with shared_ptr
  {
    std::shared_ptr<int> hostptr{new int[size], std::default_delete<int[]>{}};

    {
      s::buffer<int> buf{size};
      buf.set_final_data(hostptr);

      q.submit([&](auto &cgh) {
        auto acc = buf.get_access<s::access::mode::write>(cgh);
        cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
      });
    }

    for (int i = 0; i != size; ++i)
      BOOST_CHECK(hostptr.get()[i] == testVal);
  }

  // Test buffer construction with std::shared_ptr<T[]>
  {
    std::shared_ptr<int[]> hostptr{new int[size]};

    {
      s::buffer<int> buf{hostptr, size};

      // buffer must copy the shared_ptr, so its ref count should be two now
      BOOST_CHECK(hostptr.use_count() == 2);

      q.submit([&](auto &cgh) {
        auto acc = buf.get_access<s::access::mode::write>(cgh);
        cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
      });
    }

    for (int i = 0; i != size; ++i)
      BOOST_CHECK(hostptr.get()[i] == testVal);
  }
}

BOOST_AUTO_TEST_CASE(buffer_uninitialized) {
  namespace s = cl::sycl;

  struct bad_type {
    bad_type() {
      BOOST_ERROR(
          "The buffer dared to call the constructor that shall not be called");
    }

    // This is okay
    bad_type(int val) : val{val} {}

    int val;
  };

  const std::size_t size = 1024;
  int testVal = 42;

  { // Check "normal" uninitialized buffer
    s::buffer<bad_type> buf{size};

    s::queue{}.submit([&](auto &cgh) {
      auto acc = buf.get_access<s::access::mode::write>(cgh);
      cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
    });

    auto ha = buf.get_host_access();
    for (auto val : ha)
      BOOST_CHECK(val.val == testVal);
  }

  { // Check shared_ptr uninitialized buffer
    std::shared_ptr<bad_type> hostptr;
    s::buffer<bad_type> buf{hostptr, size};

    s::queue{}.submit([&](auto &cgh) {
      auto acc = buf.get_access<s::access::mode::write>(cgh);
      cgh.parallel_for(size, [=](auto idx) { acc[idx] = testVal; });
    });

    auto ha = buf.get_host_access();
    for (auto val : ha)
      BOOST_CHECK(val.val == testVal);
  }
}

BOOST_AUTO_TEST_SUITE_END()
