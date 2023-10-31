/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

BOOST_AUTO_TEST_SUITE_END()
