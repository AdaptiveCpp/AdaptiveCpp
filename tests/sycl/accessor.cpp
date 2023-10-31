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


#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "sycl_test_suite.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

BOOST_FIXTURE_TEST_SUITE(accessor_tests, reset_device_fixture)


BOOST_AUTO_TEST_CASE(local_accessors) {
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
          const auto lid = item.get_local_id(0);
          scratch[lid] = acc[item.get_global_id()];
          item.barrier();
          for(size_t i = local_size/2; i > 0; i /= 2) {
            if(lid < i) scratch[lid] += scratch[lid + i];
            item.barrier();
          }
          if(lid == 0) acc[item.get_global_id()] = scratch[0];
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

  cl::sycl::accessor<int, 1, mode::read_write, target::global_buffer,
                     placeholder::true_t>
      ph_acc{buf};

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(ph_acc);
    cgh.parallel_for<class placeholder_accessors1>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        ph_acc[tid] *= 2;
      });
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    auto ph_acc_copy = ph_acc; // Test that placeholder accessors can be copied
    cgh.require(ph_acc_copy);
    cgh.parallel_for<class placeholder_accessors2>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        ph_acc_copy[tid] *= 2;
      });
  });

  {
    auto acc = buf.get_access<mode::read>();
    for(size_t i = 0; i < num_elements; ++i) {
      BOOST_REQUIRE(acc[i] == 4 * i);
    }
  }
}

// TODO: Extend this
BOOST_AUTO_TEST_CASE(accessor_api) {
  namespace s = cl::sycl;

  s::buffer<int, 1> buf_a(32);
  s::buffer<int, 1> buf_b(32);
  s::buffer<int, 3> buf_d(s::range<3>{4, 4, 4});
  auto buf_c = buf_a;

  const auto run_test = [&](auto get_access) {
    auto acc_a1 = get_access(buf_a);
    auto acc_a2 = acc_a1;
    auto acc_a3 = get_access(buf_a, s::range<1>(16));
    auto acc_a4 = get_access(buf_a, s::range<1>(16), s::id<1>(4));
    auto acc_a5 = get_access(buf_a);
    auto acc_b1 = get_access(buf_b);
    auto acc_c1 = get_access(buf_c);

    BOOST_REQUIRE(acc_a1 == acc_a1);
    BOOST_REQUIRE(acc_a2 == acc_a2);
    BOOST_REQUIRE(acc_a1 != acc_a3);
    BOOST_REQUIRE(acc_a1 != acc_a4);
    BOOST_REQUIRE(acc_a1 != acc_b1);
    // NOTE: A strict reading of the 1.2.1 Rev 7 spec, section 4.3.2 would imply
    // that these should not be equal, as they are not copies of acc_a1.
    //
    // These tests are currently commented out because the expected results
    // differ between host and device accessors if the comparisons are
    // tested in command group scope instead of kernel scope.
    //BOOST_REQUIRE(acc_a1 == acc_a5);
    //BOOST_REQUIRE(acc_a1 == acc_c1);
  };

  // Test host accessors
  run_test([&](auto buf, auto... args) {
    return buf.template get_access<s::access::mode::read>(args...);
  });

  // host_accessor is default-constructible, copy-constructible, copy-assignable, equality-comparable, swappable
  s::host_accessor<int, 1> ha1;
  s::host_accessor<int, 2> ha2;
  s::host_accessor<int, 3> ha3;
  s::host_accessor<int, 1> ha1_copy = ha1;
  s::host_accessor<int, 2> ha2_copy = ha2;
  s::host_accessor<int, 3> ha3_copy = ha3;
  ha1 = ha1_copy;
  ha2 = ha2_copy;
  ha3 = ha3_copy;
  (void) (ha1 == ha1_copy);
  (void) (ha2 == ha2_copy);
  (void) (ha3 == ha3_copy);
  (void) (ha1 != ha1_copy);
  (void) (ha2 != ha2_copy);
  (void) (ha3 != ha3_copy);
  ha1.swap(ha1_copy);
  ha2.swap(ha2_copy);
  ha3.swap(ha3_copy);

  // Test device accessors
  s::queue queue;
  queue.submit([&](s::handler& cgh) {
    run_test([&](auto buf, auto... args) {
      return buf.template get_access<s::access::mode::read>(cgh, args...);
    });
    cgh.single_task<class accessor_api_device_accessors>([](){});
  });

  queue.submit([&](s::handler& cgh) {
    run_test([&](auto buf, auto... args) {
      return buf.template get_access<s::access::mode::atomic>(cgh, args...);
    });
    // mostly compilation test
    auto atomicAcc = buf_a.template get_access<s::access::mode::atomic>(cgh);
    auto atomicAcc3D = buf_d.template get_access<s::access::mode::atomic>(cgh);
    auto localAtomic = s::accessor<int, 1, s::access::mode::atomic, s::access::target::local>{s::range<1>{2}, cgh};
    auto localAtomic3D = s::accessor<int, 3, s::access::mode::atomic, s::access::target::local>{s::range<3>{2, 2, 2}, cgh};
    cgh.parallel_for<class accessor_api_atomic_device_accessors>(
        cl::sycl::nd_range<1>{2, 2},
        [=](cl::sycl::nd_item<1> item) {
          atomicAcc[0].exchange(0);
          atomicAcc3D[0][1][0].exchange(0);
          localAtomic[0].exchange(0);
          localAtomic3D[0][1][0].exchange(0);
    });
  });

  // Test local accessors
  queue.submit([&](s::handler& cgh) {
    s::accessor<int, 1, s::access::mode::read_write, s::access::target::local> acc_a(32, cgh);
    s::accessor<int, 1, s::access::mode::read_write, s::access::target::local> acc_b(32, cgh);
    auto acc_c = acc_a;

    BOOST_REQUIRE(acc_a == acc_a);
    BOOST_REQUIRE(acc_a != acc_b);
    BOOST_REQUIRE(acc_a == acc_c);

    cgh.parallel_for<class accessor_api_local_accessors>(s::nd_range<1>(1, 1), [](s::nd_item<1>){});
  });

  // local_accessor is default-constructible, copy-constructible, copy-assignable, equality-comparable, swappable
  s::local_accessor<int, 1> la1;
  s::local_accessor<int, 2> la2;
  s::local_accessor<int, 3> la3;
  s::local_accessor<int, 1> la1_copy = la1;
  s::local_accessor<int, 2> la2_copy = la2;
  s::local_accessor<int, 3> la3_copy = la3;
  la1 = la1_copy;
  la2 = la2_copy;
  la2 = la2_copy;
  (void) (la1 == la1_copy);
  (void) (la2 == la2_copy);
  (void) (la3 == la3_copy);
  (void) (la1 != la1_copy);
  (void) (la2 != la2_copy);
  (void) (la3 != la3_copy);
  la1.swap(la1_copy);
  la2.swap(la2_copy);
  la3.swap(la3_copy);
}

BOOST_AUTO_TEST_CASE(nested_subscript) {
  namespace s = cl::sycl;
  s::queue q;
  
  s::range<2> buff_size2d{64,64};
  s::range<3> buff_size3d{buff_size2d[0],buff_size2d[1],64};
  
  s::buffer<int, 2> buff2{buff_size2d};
  s::buffer<int, 3> buff3{buff_size3d};
  
  q.submit([&](s::handler& cgh){
    auto acc = buff2.get_access<s::access::mode::discard_read_write>(cgh);
    
    cgh.parallel_for<class nested_subscript2d>(buff_size2d, [=](s::id<2> idx){
      size_t x = idx[0];
      size_t y = idx[1];
      // Writing
      acc[x][y] = static_cast<int>(x*buff_size2d[1] + y);
      // Reading and making sure access id the same as with operator[id<>]
      if(acc[x][y] != acc[idx])
        acc[x][y] = -1;
    });
  });
  
  q.submit([&](s::handler& cgh){
    auto acc = buff3.get_access<s::access::mode::discard_read_write>(cgh);
    
    cgh.parallel_for<class nested_subscript3d>(buff_size3d, [=](s::id<3> idx){
      size_t x = idx[0];
      size_t y = idx[1];
      size_t z = idx[2];
      // Writing
      acc[x][y][z] = static_cast<int>(x*buff_size3d[1]*buff_size3d[2] + y*buff_size3d[2] + z);
      // Reading and making sure access id the same as with operator[id<>]
      if(acc[x][y][z] != acc[idx])
        acc[x][y][z] = -1;
    });
  });
  
  auto host_acc2d = buff2.get_access<s::access::mode::read>();
  auto host_acc3d = buff3.get_access<s::access::mode::read>();
  
  for(size_t x = 0; x < buff_size3d[0]; ++x)
    for(size_t y = 0; y < buff_size3d[1]; ++y) {
       
      size_t linear_id2d = static_cast<int>(x*buff_size2d[1] + y);
      s::id<2> id2d{x,y};
      BOOST_CHECK(host_acc2d[id2d] == linear_id2d);
      BOOST_CHECK(host_acc2d.get_pointer()[linear_id2d] == linear_id2d);
        
      for(size_t z = 0; z < buff_size3d[2]; ++z) {
        size_t linear_id3d = x*buff_size3d[1]*buff_size3d[2] + y*buff_size3d[2] + z;
        s::id<3> id3d{x,y,z};
        BOOST_CHECK(host_acc3d[id3d] == linear_id3d);
        BOOST_CHECK(host_acc3d.get_pointer()[linear_id3d] == linear_id3d);
      }
    }
}

template <class T, int Dim, cl::sycl::access_mode M, cl::sycl::target Tgt,
          cl::sycl::access::placeholder P>
constexpr cl::sycl::access_mode
get_access_mode(cl::sycl::accessor<T, Dim, M, Tgt, P>) {
  return M;
}

template <class T, int Dim, cl::sycl::access_mode M>
constexpr cl::sycl::access_mode
get_access_mode(cl::sycl::host_accessor<T, Dim, M>) {
  return M;
}

template <class T, int Dim, cl::sycl::access_mode M, cl::sycl::target Tgt,
          cl::sycl::access::placeholder P>
constexpr cl::sycl::target
get_access_target(cl::sycl::accessor<T, Dim, M, Tgt, P>) {
  return Tgt;
}

template <class Acc>
void validate_accessor_deduction(Acc acc, cl::sycl::access_mode expected_mode,
                                 cl::sycl::target expected_target) {
  BOOST_CHECK(get_access_mode(acc) == expected_mode);
  BOOST_CHECK(get_access_target(acc) == expected_target);
}

template <class Acc>
void validate_host_accessor_deduction(Acc acc,
                                      cl::sycl::access_mode expected_mode) {
  BOOST_CHECK(get_access_mode(acc) == expected_mode);
}

BOOST_AUTO_TEST_CASE(accessor_simplifications) {
  namespace s = cl::sycl;
  s::queue q;

  s::range size{1024};
  s::buffer<int> buff{size};
  s::accessor non_tagged_placeholder{buff};
  BOOST_CHECK(get_access_mode(non_tagged_placeholder)
    == s::access_mode::read_write);

  s::accessor placeholder{buff, s::read_only};
  BOOST_CHECK(placeholder.is_placeholder());

  q.submit([&](s::handler& cgh){
    s::accessor acc1{buff, cgh, s::read_only};
    BOOST_CHECK(!acc1.is_placeholder());
    
#ifdef HIPSYCL_EXT_ACCESSOR_VARIANT_DEDUCTION
    // Conversion rw accessor<int> -> accessor<const int>, read-only
    s::accessor<const int> acc2 = s::accessor<int>{buff, cgh};
    s::accessor acc3{buff, cgh, s::read_only};
    // Conversion read-write to non-const read-only accessor
    acc3 = s::accessor{buff, cgh, s::read_write};
#else
    // Conversion rw accessor<int> -> accessor<const int>, read-only
    s::accessor<const int> acc2 = s::accessor{buff, cgh, s::read_write};
    s::accessor acc3{buff, cgh, s::read_only};
    // Conversion read-write to non-const read-only accessor
    acc3 = s::accessor<int>{buff, cgh};
#endif
    BOOST_CHECK(!acc3.is_placeholder());

    // Deduction based on constness of argument
    // First employ implicit conversion to const int buff -
    // it is curently unclear whether it should also work
    // on non-const buffer, and if so, how.
    s::buffer<const int> buff2 = buff;
    s::accessor<const int> acc4{buff2, cgh};
    BOOST_CHECK(get_access_mode(acc4) == s::access_mode::read);
    s::accessor<int> acc5{buff, cgh};
    BOOST_CHECK(get_access_mode(acc5) == s::access_mode::read_write);

    // Deduction Tags
    validate_accessor_deduction(s::accessor{buff, cgh, s::read_only},
                                s::access_mode::read, s::target::device);
    validate_accessor_deduction(s::accessor{buff, cgh, s::read_only, s::no_init},
                                s::access_mode::read, s::target::device);

    validate_accessor_deduction(s::accessor{buff, cgh, s::read_write},
                                s::access_mode::read_write, s::target::device);
    validate_accessor_deduction(s::accessor{buff, cgh, s::read_write, s::no_init},
                                s::access_mode::read_write, s::target::device);

    validate_accessor_deduction(s::accessor{buff, cgh, s::write_only},
                                s::access_mode::write, s::target::device);
    validate_accessor_deduction(s::accessor{buff, cgh, s::write_only, s::no_init},
                            s::access_mode::write, s::target::device);

    validate_accessor_deduction(s::accessor{buff, cgh, s::read_only_host_task},
                                s::access_mode::read, s::target::host_task);
    validate_accessor_deduction(s::accessor{buff, cgh, s::read_only_host_task, s::no_init},
                                s::access_mode::read, s::target::host_task);

    validate_accessor_deduction(s::accessor{buff, cgh, s::read_write_host_task},
                                s::access_mode::read_write, s::target::host_task);
    validate_accessor_deduction(s::accessor{buff, cgh, s::read_write_host_task, s::no_init},
                                s::access_mode::read_write, s::target::host_task);

    validate_accessor_deduction(s::accessor{buff, cgh, s::write_only_host_task},
                                s::access_mode::write, s::target::host_task);
    validate_accessor_deduction(s::accessor{buff, cgh, s::write_only_host_task, s::no_init},
                                s::access_mode::write, s::target::host_task);

    // deduction guides without deduction tags
    validate_accessor_deduction(s::accessor{buff, cgh, s::property_list{s::no_init}},
                                s::access_mode::read_write, s::target::device);

    cgh.single_task([=](){});
  });
  {
    s::host_accessor hacc {buff};
    validate_host_accessor_deduction(hacc, s::access_mode::read_write);

    // Conversion to read-only
    s::host_accessor<const int> const_hacc = hacc;
    validate_host_accessor_deduction(const_hacc, s::access_mode::read);
  }
  {
    s::host_accessor hacc {buff, s::read_only};
    validate_host_accessor_deduction(hacc, s::access_mode::read);
  }
  {
    s::host_accessor hacc {buff, s::write_only};
    validate_host_accessor_deduction(hacc, s::access_mode::write);
  }


  q.wait();
}

BOOST_AUTO_TEST_CASE(unranged_accessor_1d_iterator) {
  namespace s = cl::sycl;

  std::array<int, 1024> host_data;
  std::iota(std::begin(host_data), std::end(host_data), 0);

  {
    s::buffer<int> buf(host_data.data(), host_data.size());

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int> acc(buf, cgh);

      cgh.single_task([=](){
        for (auto it = acc.begin(); it != acc.end(); ++it)
          *it += 1;
      });
    }).wait();
  }

  for (int i=0; i<host_data.size(); ++i)
    BOOST_CHECK_EQUAL(host_data[i], i+1);
}

BOOST_AUTO_TEST_CASE(unranged_accessor_2d_iterator) {
  namespace s = cl::sycl;

  constexpr int N = 32;
  std::array<int, N*N> host_data;
  std::iota(std::begin(host_data), std::end(host_data), 0);

  {
    s::buffer<int, 2> buf(host_data.data(), {N,N});

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int, 2> acc(buf, cgh);

      cgh.single_task([=](){
        for (auto it = acc.begin(); it != acc.end(); ++it)
          *it += 1;
      });
    }).wait();
  }

  for (int i=0; i<host_data.size(); ++i)
    BOOST_CHECK_EQUAL(host_data[i], i+1);
}

BOOST_AUTO_TEST_CASE(unranged_accessor_3d_iterator) {
  namespace s = cl::sycl;

  constexpr int N = 3;
  std::array<int, N*N*N> host_data;
  std::iota(std::begin(host_data), std::end(host_data), 0);

  // Count iterations of for loop below to check if iterator stays within bounds
  int it_counter = 0; 
  {
    s::buffer<int, 3> buf(host_data.data(), {N,N,N});
    s::buffer<int, 1> it_buf(&it_counter, 1);

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int, 3> acc(buf, cgh);
      s::accessor<int, 1> it_acc(it_buf, cgh);

      cgh.single_task([=](){
        for (auto &it : acc) {
          it += 1;
          it_acc[0]++;
        }
      });
    }).wait();
  }

  for (int i=0; i<host_data.size(); ++i)
    BOOST_CHECK_EQUAL(host_data[i], i+1);

  BOOST_CHECK_EQUAL(it_counter, N*N*N);
}

BOOST_AUTO_TEST_CASE(ranged_accessor_1d_iterator) {
  namespace s = cl::sycl;

  constexpr int N = 1024;
  const s::range range(512);
  const s::id offset(10);
  
  std::array<int, N> host_data;
  std::iota(std::begin(host_data), std::end(host_data), 0);
  
  {
    s::buffer<int> buf(host_data.data(), N);

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int> acc(buf, cgh, range, offset);

      cgh.single_task([=](){
        for (auto it = acc.begin(); it != acc.end(); ++it)
          *it = -1;
      });
    }).wait();
  }

  for (int i=0; i < offset[0]; ++i)
    BOOST_CHECK_EQUAL(host_data[i], i);
  for (int i = offset[0]; i < offset[0] + range[0]; ++i)
    BOOST_CHECK_EQUAL(host_data[i], -1);
  for (int i = offset[0] + range[0]; i<N; ++i)
    BOOST_CHECK_EQUAL(host_data[i], i);
}

BOOST_AUTO_TEST_CASE(ranged_accessor_2d_iterator) {
  namespace s = cl::sycl;

  constexpr int N1 = 32;
  constexpr int N2 = 64;
  const s::range<2> range{2, 4};
  const s::id<2> offset{2, 5};
  
  std::array<int, N1*N2> host_data;
  std::fill(std::begin(host_data), std::end(host_data), 0);
  
  {
    s::buffer<int, 2> buf(host_data.data(), {N1,N2});

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int, 2> acc(buf, cgh, range, offset);

      cgh.single_task([=](){
        for (auto it = acc.begin(); it < acc.end(); ++it)
          *it = 1;
      });
    }).wait();
  }

  for (int i=0; i<N1; ++i) {
    for (int j=0; j<N2; ++j) {
      if ((i >= offset[0]) &&
          (i < offset[0] + range[0]) &&
          (j >= offset[1]) &&
          (j < offset[1] + range[1]))
        BOOST_CHECK_EQUAL(host_data[i*N2+j], 1);
      else
        BOOST_CHECK_EQUAL(host_data[i*N2+j], 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(ranged_accessor_3d_iterator) {
  namespace s = cl::sycl;

  constexpr int N1 = 8;
  constexpr int N2 = 16;
  constexpr int N3 = 32;

  const s::range<3> range{2, 2, 2};
  const s::id<3> offset{1, 2, 0};
  
  std::array<int, N1*N2*N3> host_data;
  std::fill(std::begin(host_data), std::end(host_data), 0);
  
  {
    s::buffer<int, 3> buf(host_data.data(), {N1,N2,N3});

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int, 3> acc(buf, cgh, range, offset);

      cgh.single_task([=](){
        for (auto it = acc.begin(); it < acc.end(); ++it)
          *it = 1;
      });
    }).wait();
  }

  for (int i=0; i<N1; ++i) {
    for (int j=0; j<N2; ++j) {
      for (int k=0; k<N3; ++k) {
        if ((i >= offset[0]) &&
            (i < offset[0] + range[0]) &&
            (j >= offset[1]) &&
            (j < offset[1] + range[1]) &&
            (k >= offset[2]) &&
            (k < offset[2] + range[2]))
          BOOST_CHECK_EQUAL(host_data[i*N2*N3 + j*N3 + k], 1);
        else
          BOOST_CHECK_EQUAL(host_data[i*N2*N3 + j*N3 + k], 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(reverse_iterator) {
  namespace s = cl::sycl;

  std::array<int, 1024> host_data;
  std::iota(std::begin(host_data), std::end(host_data), 0);

  {
    s::buffer<int> buf(host_data.data(), host_data.size());

    s::queue{}.submit([&](s::handler &cgh) {
      s::accessor<int> acc(buf, cgh);

      cgh.single_task([=](){
        for (auto it = acc.rbegin(); it != acc.rend(); ++it)
          *it += 1;
      });
    }).wait();
  }

  for (int i=0; i<host_data.size(); ++i)
    BOOST_CHECK_EQUAL(host_data[i], i+1);
}

BOOST_AUTO_TEST_CASE(host_accessor_iterator) {
  namespace s = cl::sycl;

  constexpr int N = 1024;
  std::array<int, N> data;

  s::buffer<int> buf{data.data(), data.size()};
  s::host_accessor ha{buf};

  std::iota(ha.begin(), ha.end(), 0);

  for(int i=0; i<N; ++i)
    BOOST_CHECK_EQUAL(data[i], i);

  std::iota(ha.rbegin(), ha.rend(), 0);
  for(int i=0; i<N; ++i)
    BOOST_CHECK_EQUAL(data[i], N-i-1);
}

BOOST_AUTO_TEST_CASE(accessor_iterator_api) {
  namespace s = cl::sycl;

  constexpr int N = 1024;
  std::array<int, N> data;
  std::fill(data.begin(), data.end(), 0);

  s::buffer<int> buf{data.data(), data.size()};
  s::host_accessor ha{buf};

  /*** Test postfix ++ ***/
  {
    for (auto it = ha.begin(); it != ha.end(); it++)
      *it = 2;

    for(const auto& entry : data)
      BOOST_CHECK_EQUAL(entry, 2);
  }

  /*** Test operator+= ***/
  {
    std::fill(data.begin(), data.end(), 0);
    for (auto it = ha.begin(); it != ha.end(); it += 2)
      *it = 1;

    for(int i=0; i<N; ++i) {
      if (i % 2 != 0)
        BOOST_CHECK_EQUAL(data[i], 0);
      else
        BOOST_CHECK_EQUAL(data[i], 1);
    }
  }

  /*** Test prefix -- ***/
  {
    std::fill(data.begin(), data.end(), 0);
    for (auto it = ha.end() - 1; it != ha.begin() - 1; --it)
      *it = 1;

    for(const auto& entry : data)
      BOOST_CHECK_EQUAL(entry, 1);
  }

  /*** Test postfix -- ***/
  {
    std::fill(data.begin(), data.end(), 0);
    for (auto it = ha.end() - 1; it != ha.begin() - 1; it--)
      *it = 1;

    for(const auto& entry : data)
      BOOST_CHECK_EQUAL(entry, 1);
  }

  /*** Check that operator+ is commutative -- ***/
  {
    auto it = ha.begin();
    BOOST_CHECK((it + 2) == (2 + it));
  }

  /*** Test operator[] ***/
  {
    std::fill(data.begin(), data.end(), 0);

    auto it = ha.begin();
    for (int i=0; i<N; i+=2)
      it[i] = 1;

    for(int i=0; i<N; ++i) {
      if (i % 2 != 0)
        BOOST_CHECK_EQUAL(data[i], 0);
      else
        BOOST_CHECK_EQUAL(data[i], 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(offset_1d) {
  namespace s = cl::sycl;

  constexpr int N = 1024;
  std::vector<int> data(N, 1);

  {
    s::buffer<int> buf(data.data(), N);
    s::queue{}.submit([&](s::handler &cgh) {
      s::id offset{2};
      s::range range{N-offset};
      
      auto acc = s::accessor(buf, cgh, range, offset);

      cgh.parallel_for(s::range{N - offset},
                       [=](auto &idx) {
                         acc[idx] = 2;
                       });
    });
  }

  // Expected result is [1, 1, 2, 2, ..., 2]
  std::vector<int> expected(N, 2);
  expected[0] = 1;
  expected[1] = 1;

  BOOST_CHECK(expected == data);
}

BOOST_AUTO_TEST_CASE(offset_2d) {
  namespace s = cl::sycl;

  constexpr int N = 8;
  std::array<int, N*N> data;
  std::fill(data.begin(), data.end(), 1);

  {
    s::buffer<int, 2> buf(data.data(), {N,N});

    s::queue{}.submit([&](s::handler &cgh) {
      std::size_t offset_1d = 2;
      s::range range{N - offset_1d, N - offset_1d};
      s::id offset{offset_1d, offset_1d};
      
      auto acc = s::accessor(buf, cgh, range, offset);

      cgh.parallel_for(s::range{N - offset.get(0), N - offset.get(1)},
                       [=](auto &idx) {
                         acc[idx] = 2;
                       });
    }).wait();
  }

  /* Expected result is
  [ [1, 1, 1, 1, ..., 1],
    [1, 1, 1, 1, ..., 1]
    [1, 1, 2, 2, ..., 2]
    [1, 1, 2, 2, ..., 2]
    ...
    [1, 1, 2, 2, ..., 2]] */
  std::array<int,N*N> expected;
  std::fill(expected.begin(), expected.end(), 2);
  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      if ((i < 2) or (j < 2))
        expected[i*N+j] = 1;

  BOOST_CHECK(data == expected);
}

BOOST_AUTO_TEST_CASE(offset_nested_subscript) {
  namespace s = cl::sycl;

  constexpr int N = 8;
  std::array<int, N*N> data;
  std::fill(data.begin(), data.end(), 1);

  {
    s::buffer<int, 2> buf(data.data(), {N,N});
    s::queue{}.submit([&](s::handler &cgh) {
      std::size_t offset_1d = 2;
      s::range range{N - offset_1d, N - offset_1d};
      s::id offset{offset_1d, offset_1d};
      
      auto acc = s::accessor(buf, cgh, range, offset);

      cgh.parallel_for(s::range{N - offset.get(0), N - offset.get(1)},
                       [=](auto &idx) {
                         auto row = idx[0];
                         auto col = idx[1];
                         acc[row][col] = 2;
                       });
    }).wait();
  }

  /* Expected result is
  [ [1, 1, 1, 1, ..., 1],
    [1, 1, 1, 1, ..., 1]
    [1, 1, 2, 2, ..., 2]
    [1, 1, 2, 2, ..., 2]
    ...
    [1, 1, 2, 2, ..., 2]] */
  std::array<int,N*N> expected;
  std::fill(expected.begin(), expected.end(), 2);
  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      if ((i < 2) or (j < 2))
        expected[i*N+j] = 1;

  BOOST_CHECK(data == expected);
}

BOOST_AUTO_TEST_SUITE_END()
