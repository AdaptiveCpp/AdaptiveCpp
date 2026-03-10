// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include <iostream>
#include <array>
#include <sycl/sycl.hpp>
#include "common.hpp"

struct Plain {
  int x;
  int y;
};

struct Inner {
  int value;
};

struct WithNested {
  Inner inner;
  int extra;
};

struct WithArrayOfStructs {
  std::array<Inner, 3> items;
  int count;
};

struct WithCArrayOfStructs {
  Inner items[3];
  int count;
};

void test_plain(sycl::queue& q) {
  Plain* out = sycl::malloc_shared<Plain>(1, q);
  q.single_task([=]() {
    Plain p;
    p.x = 10;
    p.y = 20;
    *out = p;
  }).wait();
  // CHECK: 10
  // CHECK: 20
  std::cout << out->x << "\n" << out->y << "\n";
  sycl::free(out, q);
}

void test_nested(sycl::queue& q) {
  WithNested* out = sycl::malloc_shared<WithNested>(1, q);
  q.single_task([=]() {
    WithNested s;
    s.inner.value = 42;
    s.extra = 7;
    *out = s;
  }).wait();
  // CHECK: 42
  // CHECK: 7
  std::cout << out->inner.value << "\n" << out->extra << "\n";
  sycl::free(out, q);
}

void test_array_of_structs(sycl::queue& q) {
  WithArrayOfStructs* out = sycl::malloc_shared<WithArrayOfStructs>(1, q);
  q.single_task([=]() {
    WithArrayOfStructs s;
    s.items[0].value = 1;
    s.items[1].value = 2;
    s.items[2].value = 3;
    s.count = 3;
    *out = s;
  }).wait();
  // CHECK: 1
  // CHECK: 2
  // CHECK: 3
  // CHECK: 3
  for (int i = 0; i < out->count; ++i)
    std::cout << out->items[i].value << "\n";
  std::cout << out->count << "\n";
  sycl::free(out, q);
}

void test_carray_of_structs(sycl::queue& q) {
  WithCArrayOfStructs* out = sycl::malloc_shared<WithCArrayOfStructs>(1, q);
  q.single_task([=]() {
    WithCArrayOfStructs s;
    s.items[0].value = 4;
    s.items[1].value = 5;
    s.items[2].value = 6;
    s.count = 3;
    *out = s;
  }).wait();
  // CHECK: 4
  // CHECK: 5
  // CHECK: 6
  // CHECK: 3
  for (int i = 0; i < out->count; ++i)
    std::cout << out->items[i].value << "\n";
  std::cout << out->count << "\n";
  sycl::free(out, q);
}

int main() {
  sycl::queue q = get_queue();
  test_plain(q);
  test_nested(q);
  test_array_of_structs(q);
  test_carray_of_structs(q);
}
