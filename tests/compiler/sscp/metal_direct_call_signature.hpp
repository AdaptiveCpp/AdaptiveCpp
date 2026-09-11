#pragma once

#include <sycl/sycl.hpp>

#include <utility>

template <class T>
__attribute__((noinline)) std::pair<T, T> makeAggregatePair(T Value) {
  return {Value + T{1}, Value + T{2}};
}

SYCL_EXTERNAL __attribute__((noinline)) float
consumeAggregatePair(float Value);
