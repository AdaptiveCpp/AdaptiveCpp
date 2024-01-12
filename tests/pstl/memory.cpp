#include <algorithm>
#include <execution>
#include <vector>
#include <cstdlib>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_memory, enable_unified_shared_memory)

BOOST_AUTO_TEST_CASE(pstl_std_malloc_free) {
  constexpr std::size_t size = 64;
  constexpr int value = 42;

  int *p1 = reinterpret_cast<int *>(std::malloc(size * sizeof(int)));

  // This would fail if p1 is not a correct USM pointer
  std::fill(std::execution::par_unseq, p1, p1 + size, value);

  std::vector<int> host_data(size);
  std::fill(host_data.begin(), host_data.end(), value);

  for (std::size_t i = 0; i < size; ++i)
    BOOST_CHECK_EQUAL(host_data[i], p1[i]);

  std::free(p1);
}

BOOST_AUTO_TEST_CASE(pstl_global_malloc_free) {
  constexpr std::size_t size = 64;
  constexpr int value = 42;

  int *p1 = reinterpret_cast<int *>(malloc(size * sizeof(int)));

  // This would fail if p1 is not a correct USM pointer
  std::fill(std::execution::par_unseq, p1, p1 + size, value);

  std::vector<int> host_data(size);
  std::fill(host_data.begin(), host_data.end(), value);

  for (std::size_t i = 0; i < size; ++i)
    BOOST_CHECK_EQUAL(host_data[i], p1[i]);

  free(p1);
}

BOOST_AUTO_TEST_CASE(pstl_aligned_alloc) {
  constexpr std::size_t alignment = 8;
  constexpr std::size_t size = alignment * 8;
  constexpr int value = 42;

  int *p1 = reinterpret_cast<int *>(std::aligned_alloc(alignment, size*sizeof(int)));

  std::fill(std::execution::par_unseq, p1, p1 + size, value);

  std::vector<int> host_data(size);
  std::fill(host_data.begin(), host_data.end(), value);

  for(std::size_t i = 0; i < size; ++i)
    BOOST_CHECK_EQUAL(host_data[i], p1[i]);

  std::free(p1);
}

BOOST_AUTO_TEST_SUITE_END()
