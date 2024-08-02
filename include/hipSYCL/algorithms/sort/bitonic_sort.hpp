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

#ifndef ACPP_ALGORITHMS_BITONIC_SORT
#define ACPP_ALGORITHMS_BITONIC_SORT

#include <iterator>
#include "hipSYCL/sycl/queue.hpp"

namespace hipsycl::algorithms::sorting {


// This function is based on the code from SyclParallelSTL, and subject
// to the following copyright and license:
/*
Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
  */
/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <class RandomIt, class Comparator>
sycl::event bitonic_sort(sycl::queue &q, RandomIt first, RandomIt last,
                         Comparator comp) {

  std::size_t num_elements = std::distance(first, last);

  int num_stages = 0;
  // 2^numStages should be equal to length
  // i.e number of times you halve the lenght to get 1 should be numStages
  for (int tmp = num_elements; tmp > 1; tmp >>= 1) {
    ++num_stages;
  }

  sycl::event most_recent_event;
  sycl::range<1> r{num_elements / 2};

  using T = typename std::iterator_traits<RandomIt>::value_type;

  for (int stage = 0; stage < num_stages; ++stage) {
    // Every stage has stage + 1 passes
    for (int pass = 0; pass < stage + 1; ++pass) {

      auto advance_to = [](RandomIt first, auto i) -> RandomIt {
        std::advance(first, i);
        return first;
      };

      auto kernel = [=](sycl::id<1> idx) {

        int sort_increasing = 1;
        
        std::size_t gid = idx.get(0);

        int pair_distance = 1 << (stage - pass);
        int block_width = 2 * pair_distance;

        std::size_t left_id =
            (gid % pair_distance) + (gid / pair_distance) * block_width;
        std::size_t right_id = left_id + pair_distance;

        T left_element = *advance_to(first, left_id);
        T right_element = *advance_to(first, right_id);

        std::size_t same_direction_block_width = 1 << stage;

        if ((gid / same_direction_block_width) % 2 == 1) {
          sort_increasing = 1 - sort_increasing;
        }

        T greater = left_element;
        T lesser = right_element;

        if (comp(left_element, right_element)) {
          greater = right_element;
          lesser = left_element;
        } else {
          greater = left_element;
          lesser = right_element;
        }

        *advance_to(first, left_id) = sort_increasing ? lesser : greater;
        *advance_to(first, right_id) = sort_increasing ? greater : lesser;
      };

      if((stage == 0 && pass == 0) || q.is_in_order())
        most_recent_event = q.parallel_for(r, kernel);
      else
        most_recent_event = q.parallel_for(r, most_recent_event, kernel);

    } // pass_stage
  }   // stage
  return most_recent_event;
} // bitonic_sort
}

#endif
