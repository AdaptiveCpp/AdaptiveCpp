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

#include <initializer_list>

#include <CL/sycl.hpp>

// What we're testing here is different forms of function calls, including
// implicit and templated functions and constructors as well as destructors.
// The hipSYCL Clang plugin must recognize that each of these is being called
// by the user-provided kernel functor, and mark them as __device__ functions
// accordingly.

void foo4(int v) {
  printf("foo4: %d\n", v);
}

struct Conversion {
  Conversion(float value) : value(value) {
    foo4(*this);
  }

  operator int() {
    return static_cast<int>(value);
  }

  float value;
};

void foo3(int v) {
  printf("foo3: %d\n", v);
}

void foo2(Conversion foo) {
  printf("foo2: %f\n", foo.value);
}

void foo1(int v) {
  printf("foo1: %d\n", v + 5);
  ([=](){ foo3(v); })();
}

struct MyParent {
  MyParent(std::initializer_list<int> init) {
    for(auto&& e : init) {
      parent_value += e;
    }
    foo1(parent_value);
  }

  virtual ~MyParent() {
    printf("~MyParent: %d\n", parent_value);
  };

  int parent_value = 0;
};

struct MyStruct : MyParent {
  MyStruct(int value) : MyParent::MyParent({value, 4, 5, 7}),
    value(value) {
  }

  ~MyStruct() {
    foo2(static_cast<float>(value));
  }

  int value;
};

template<typename T>
int template_fn() {
  T t(42);
  return t.value;
}

float some_fn() {
  return template_fn<MyStruct>();
}

int main() {
  cl::sycl::queue queue;
  cl::sycl::buffer<float, 1> buf(10);

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class some_kernel>(buf.get_range(), [=](cl::sycl::item<1> item) {
        acc[item] = some_fn();
      });
  });

  return 0;
}

