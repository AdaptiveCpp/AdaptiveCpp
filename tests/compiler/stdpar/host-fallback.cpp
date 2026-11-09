// RUN: %acpp %s --acpp-targets=generic -std=c++17 -O2 -I%S/../../../include \
// RUN:   -D_GLIBCXX_USE_TBB_PAR_BACKEND=0 -o %t
// RUN: %t
// RUN: %t exception
// RUN: %t bad_alloc

// Test the host adapter without SYCL or allocation interception. Use GNU's
// serial PSTL backend so this test does not require a TBB link dependency.
#include <algorithm>
#include <execution>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <new>

#include <hipSYCL/std/stdpar/detail/execution_fwd.hpp>
#include <hipSYCL/std/stdpar/detail/stdpar_defs.hpp>

#define HOST_FALLBACK(algorithm, ...) \
  HIPSYCL_STDPAR_HOST_FALLBACK(algorithm, hipsycl::stdpar::par_unseq_host_fallback, __VA_ARGS__)

int main(int argc, char** argv) {
  int data[] = {1, 2, 3, 4};
  int output[4] = {};

  if(argc > 1) {
    // A user-thrown bad_alloc must terminate too, not be mistaken for failure
    // to allocate the parallel algorithm's own temporary storage.
    std::set_terminate([] { std::_Exit(0); });
    try {
      HOST_FALLBACK(for_each, data, data + 4, [&](int&) {
        if(argv[1][0] == 'b')
          throw std::bad_alloc{};
        throw 42;
      });
    } catch(...) {
      return 1;
    }
    return 2;
  }

  HOST_FALLBACK(for_each, data, data + 4, [](int& x) { ++x; });
  assert(data[0] == 2 && data[3] == 5);
  auto end = HOST_FALLBACK(for_each_n, data, 4, [](int& x) { --x; });
  assert(end == data + 4 && data[0] == 1 && data[3] == 4);
  assert(HOST_FALLBACK(for_each_n, data, 0, [](int& x) { ++x; }) == data);
#ifdef _LIBCPP_VERSION
  assert(HOST_FALLBACK(for_each_n, data, -1, [](int& x) { ++x; }) == data);
#endif

  end = HOST_FALLBACK(fill_n, output, 4, 7);
  assert(end == output + 4 && output[0] == 7 && output[3] == 7);
  assert(HOST_FALLBACK(fill_n, output, 0, 9) == output);
  end = HOST_FALLBACK(generate_n, output, 4, [] { return 3; });
  assert(end == output + 4 && output[3] == 3);

  end = HOST_FALLBACK(copy_if, data, data + 4, output,
                      [](int x) { return x % 2 == 0; });
  assert(end == output + 2 && output[0] == 2 && output[1] == 4);
  end = HOST_FALLBACK(transform, data, data + 4, output,
                      [](int x) { return x * 2; });
  assert(end == output + 4 && output[0] == 2 && output[3] == 8);

  assert(HOST_FALLBACK(reduce, data, data + 4, 0) == 10);
  assert(HOST_FALLBACK(reduce, data, data, 17) == 17);
  assert(HOST_FALLBACK(transform_reduce, data, data + 4, data, 0) == 30);
  end = HOST_FALLBACK(inclusive_scan, data, data + 4, output);
  assert(end == output + 4 && output[0] == 1 && output[3] == 10);
  end = HOST_FALLBACK(exclusive_scan, data, data + 4, output, 0);
  assert(end == output + 4 && output[0] == 0 && output[3] == 6);

  auto mismatch = HOST_FALLBACK(mismatch, data, data + 4, data);
  assert(mismatch.first == data + 4 && mismatch.second == data + 4);
  HOST_FALLBACK(reverse, data, data + 4);
  assert(data[0] == 4 && data[3] == 1);
  HOST_FALLBACK(sort, data, data + 4);
  assert(data[0] == 1 && data[3] == 4);
}
