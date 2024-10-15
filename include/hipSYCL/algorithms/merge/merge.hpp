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

#ifndef ACPP_ALGORITHMS_MERGE_HPP
#define ACPP_ALGORITHMS_MERGE_HPP

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

#include "merge_path.hpp"

namespace hipsycl::algorithms::merging {

namespace detail {

template <class ForwardIt1, class ForwardIt2, class OutputIt, class Compare, class Size>
void sequential_merge(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                      ForwardIt2 last2, OutputIt out, Compare comp, Size max_num_merged) {

  auto initial_out = out;
  auto copy_remaining = [&](auto first, auto last) {
    for (; first != last && (std::distance(initial_out, out) < max_num_merged);
         ++first, ++out)
      *out = *first;
  };

  for (; first1 != last1 && (std::distance(initial_out, out) < max_num_merged);
       ++out) {
    if(first2 == last2) {
      copy_remaining(first1, last1);
      return;
    } else {
      auto f1 = *first1;
      auto f2 = *first2;
      if(comp(f1, f2)) {
        *out = f1;
        ++first1;
      } else {
        *out = f2;
        ++first2;
      }
    }
  }
  copy_remaining(first2, last2);
}


/// Decomposes the problem into N independent merges of given size, and
/// then runs sequential merge on them. This might be a good strategy on CPU.
///
/// Precondition: distance(fist1, last1) > 0 && distance(first2, last2) > 0.
/// Otherwise we cannot run the merge path algorithm for decomposing the merge.
template <class RandomIt1, class RandomIt2, class OutputIt, class Compare>
void segmented_merge(
    RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2,
    OutputIt out, Compare comp, std::size_t partition_index,
    std::size_t partition_chunk_size) {

  std::size_t p1 = 0;
  std::size_t p2 = 0;

  merge_path::nth_independent_merge_begin(first1, last1, first2, last2, comp,
                                          partition_index,
                                          partition_chunk_size, p1, p2);

  auto chunk_first1 = first1;
  auto chunk_first2 = first2;

  std::advance(chunk_first1, p1);
  std::advance(chunk_first2, p2);

  auto chunk_last1 = chunk_first1;
  auto chunk_last2 = chunk_first2;

  std::advance(chunk_last1, std::min(partition_chunk_size,
                              std::distance(first1, last1) - p1));
  std::advance(chunk_last2, std::min(partition_chunk_size,
                              std::distance(first2, last2) - p2));

  std::size_t chunk_out_offset = partition_index * partition_chunk_size;
  auto chunk_out = out;
  std::advance(chunk_out, chunk_out_offset);

  sequential_merge(chunk_first1, chunk_last1, chunk_first2, chunk_last2,
                    chunk_out, comp, partition_chunk_size);
}

}

/// Precondition: distance(fist1, last1) > 0 && distance(first2, last2) > 0.
/// Otherwise we cannot run the merge path algorithm for decomposing the merge.
template <class RandomIt1, class RandomIt2, class OutputIt, class Compare>
sycl::event segmented_merge(sycl::queue &q, RandomIt1 first1, RandomIt1 last1,
                            RandomIt2 first2, RandomIt2 last2, OutputIt out,
                            Compare comp,
                            std::size_t partition_chunk_size = 128,
                            const std::vector<sycl::event> &deps = {}) {

  //detail::print_merge_matrix(first1, last1, first2, last2, comp);

  std::size_t problem_size = merge_path::num_partitions(
      first1, last1, first2, last2, partition_chunk_size);

  if(problem_size == 0)
    return sycl::event{};

  return q.parallel_for(sycl::range{problem_size}, deps, [=](sycl::id<1> idx) {
    detail::segmented_merge(first1, last1, first2, last2, out, comp, idx.get(0),
                            partition_chunk_size);
  });
}
}





#endif
