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

#include "unit_tests.hpp"

#include <initializer_list>

// NOTE: Since these tests are concerned with device compilation, the first
// and arguably most important "test" is that this file compiles at all. We
// then also include a few unit tests to assert the correct behavior of the
// resulting programs.

BOOST_FIXTURE_TEST_SUITE(device_compilation_test_suite, reset_device_fixture)

int add_five(int v) {
  return v + 5;
}

void call_lambda(int& v) {
  ([&](){ v = add_five(v); })();
}

struct MyParent {
  MyParent(std::initializer_list<int> init) {
    for(auto&& e : init) {
      parent_value += e;
    }
    call_lambda(parent_value);
  }

  virtual ~MyParent() {
    *increment_in_dtor += parent_value + 7;
  };

  int parent_value = 0;
  int* increment_in_dtor;
};

struct Conversion {
  Conversion(float value) : value(value) {}

  operator int() {
    return static_cast<int>(value);
  }

  float value;
};

int convert(Conversion conv) {
  return add_five(conv);
}

struct MyStruct : MyParent {
  MyStruct(int& value) : MyParent::MyParent{value, 4, 5, 7}, value(value) {
    increment_in_dtor = &value;
  }

  ~MyStruct() {
    value = convert(static_cast<float>(value));
  }

  int& value;
};

template<typename T, typename U>
void template_fn(U& u) {
  T t(u);
}

float some_fn(int value) {
  template_fn<MyStruct>(value);
  return value;
}

// What we're testing here is different forms of function calls, including
// implicit and templated functions and constructors as well as destructors.
// The hipSYCL Clang plugin must recognize that each of these is being called
// by the user-provided kernel functor, and mark them as __device__ functions
// accordingly.
BOOST_AUTO_TEST_CASE(complex_device_call_graph) {
  cl::sycl::queue queue;
  cl::sycl::buffer<float, 1> buf(1);

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class some_kernel>([=]() {
        acc[0] = some_fn(2);
      });
  });

  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  BOOST_REQUIRE(acc[0] == 37);
}


template<int T, int* I, typename U>
class complex_kn;

template<typename T>
class templated_kn;

enum class my_enum { HELLO = 42, WORLD };

template<my_enum E>
class enum_kn;

template<template <typename, int, typename> class T>
class templated_template_kn;

BOOST_AUTO_TEST_CASE(kernel_name_mangling) {
  cl::sycl::queue queue;

  // Qualified / modified types
  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<const unsigned int>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<complex_kn<32, nullptr, enum_kn<my_enum::HELLO>>>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_template_kn<cl::sycl::buffer>>
        ([](){});
  });

  // No collision happens between the following two names
  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<class collision>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <class templated_kn_collision>
        ([](){});
  });
}

struct KernelFunctor {
  using Accessor = cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::discard_write>;
  KernelFunctor(Accessor acc) : acc(acc) {};

  void operator()(cl::sycl::item<1> item) {
    acc[0] = 300 + item.get_linear_id();
  }

  Accessor acc;
};

// It's allowed to omit the name if the functor is globally visible
BOOST_AUTO_TEST_CASE(omit_kernel_name) {
  cl::sycl::queue queue;
  cl::sycl::buffer<size_t, 1> buf(1);

  {
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::range<1>(1), KernelFunctor(acc));
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 300);
  }

  {
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::range<1>(1), cl::sycl::id<1>(1), KernelFunctor(acc));
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 301);
  }
}

BOOST_AUTO_TEST_CASE(hierarchical_invoke_shared_memory) {
  cl::sycl::queue queue;

  // The basic case, as outlined in the SYCL spec.
  {
    cl::sycl::buffer<size_t, 1> buf(4);
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for_work_group<class shmem_one>(
        cl::sycl::range<1>(2), cl::sycl::range<1>(2), [=](cl::sycl::group<1> group) {
          { // Do this in a block to make sure the Clang plugin handles this correctly.
            size_t shmem[2]; // Should be shared
            group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
              shmem[h_item.get_local().get_linear_id()] = h_item.get_global().get_linear_id();
            });
            group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
              if(h_item.get_local().get_linear_id() == 0) {
                auto offset = h_item.get_global().get_linear_id();
                acc[offset + 0] = shmem[0];
                acc[offset + 1] = shmem[1];
              }
            });
          }
        });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    for(size_t i = 0; i < 4; ++i) {
      BOOST_REQUIRE(acc[i] == i);
    }
  }

  // In this case the functionality remains the same, but
  // is moved into a separate function. We expect it to still
  // behave as before.
  {
    auto operate_on_shmem = [](cl::sycl::group<1> group,
      cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::discard_write> acc) {
      size_t shmem[2]; // Should be shared
      group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
        shmem[h_item.get_local().get_linear_id()] = h_item.get_global().get_linear_id();
      });
      group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
        if(h_item.get_local().get_linear_id() == 0) {
          auto offset = h_item.get_global().get_linear_id();
          acc[offset + 0] = shmem[0];
          acc[offset + 1] = shmem[1];
        }
      });
    };

    cl::sycl::buffer<size_t, 1> buf(4);
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for_work_group<class shmem_two>(
        cl::sycl::range<1>(2), cl::sycl::range<1>(2), [=](cl::sycl::group<1> group) {
          operate_on_shmem(group, acc);
        });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    for(size_t i = 0; i < 4; ++i) {
      BOOST_REQUIRE(acc[i] == i);
    }
  }
}


void forward_declared2();

template<class T>
void forward_declared1();

template<class T>
class forward_declared_test_kernel;

BOOST_AUTO_TEST_CASE(forward_declared_function) {
  cl::sycl::queue q;

  // Here, the clang plugin must build the correct call graph
  // and emit both forward_declared1() and forward_declared2()
  // to device code, even though the call expression below refers
  // to the forward declarations above.
  q.submit([&](cl::sycl::handler& cgh){
    cgh.single_task<forward_declared_test_kernel<int>>([=](){
      forward_declared1<int>();
    });  
  });

  q.wait_and_throw();
}


template <class T>
void forward_declared1(){forward_declared2();}

void forward_declared2(){}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line

