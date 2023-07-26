/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_ALGORITHMS_ALGORITHM_HPP
#define HIPSYCL_ALGORITHMS_ALGORITHM_HPP

#include <iterator>
#include <limits>
#include "hipSYCL/sycl/libkernel/functional.hpp"
#include "hipSYCL/sycl/sycl.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

namespace hipsycl::algorithms {

template <class ForwardIt, class UnaryFunction2>
sycl::event for_each(sycl::queue &q, ForwardIt first, ForwardIt last,
                     UnaryFunction2 f) {

  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          f(*it);
                        });
}

template<class ForwardIt, class Size, class UnaryFunction2>
sycl::event for_each_n(sycl::queue& q,
                    ForwardIt first, Size n, UnaryFunction2 f) {
  if(n <= 0)
    // sycl::event{} represents a no-op that is always finished.
    // This means it does not respect prior tasks in the task graph!
    // TODO Is this okay? Can we defer this responsibility to the user?
    return sycl::event{};
  return q.parallel_for(sycl::range{static_cast<size_t>(n)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          f(*it);
                        });
}

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
sycl::event transform(sycl::queue& q,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first,
                     UnaryOperation unary_op) {
  return q.parallel_for(sycl::range{std::distance(first1, last1)},
                        [=](sycl::id<1> id) {
                          auto input = first1;
                          auto output = d_first;
                          std::advance(input, id[0]);
                          std::advance(output, id[0]);
                          *output = unary_op(*input);
                        });
}

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
sycl::event transform(sycl::queue &q, ForwardIt1 first1, ForwardIt1 last1,
                      ForwardIt2 first2, ForwardIt3 d_first,
                      BinaryOperation binary_op) {
  return q.parallel_for(sycl::range{std::distance(first1, last1)},
                        [=](sycl::id<1> id) {
                          auto input1 = first1;
                          auto input2 = first2;
                          auto output = d_first;
                          std::advance(input1, id[0]);
                          std::advance(input2, id[0]);
                          std::advance(output, id[0]);
                          *output = binary_op(*input1, *input2);
                        });
}

template <class ForwardIt1, class ForwardIt2>
sycl::event copy(sycl::queue &q, ForwardIt1 first, ForwardIt1 last,
                 ForwardIt2 d_first) {
  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto input = first;
                          auto output = d_first;
                          std::advance(input, id[0]);
                          std::advance(output, id[0]);
                          *output = *input;
                        });
}


template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
sycl::event copy_if(sycl::queue& q,
                    ForwardIt1 first, ForwardIt1 last,
                    ForwardIt2 d_first,
                    UnaryPredicate pred) {
  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto input = first;
                          auto output = d_first;
                          std::advance(input, id[0]);
                          std::advance(output, id[0]);
                          auto input_v = *input;
                          if(pred(input_v))
                            *output = input_v;
                        });
}

template<class ForwardIt1, class Size, class ForwardIt2 >
sycl::event copy_n(sycl::queue& q, ForwardIt1 first, Size count, ForwardIt2 result) {
  if(count <= 0)
    return sycl::event{};
  return q.parallel_for(sycl::range{static_cast<size_t>(count)},
                        [=](sycl::id<1> id) {
                          auto input = first;
                          auto output = result;
                          std::advance(input, id[0]);
                          std::advance(output, id[0]);
                          *output = *input;
                        });
}

template <class ForwardIt, class T>
sycl::event fill(sycl::queue &q, ForwardIt first, ForwardIt last,
                 const T &value) {
  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          *it = value;
                        });
}

template<class ForwardIt, class Size, class T >
sycl::event fill_n(sycl::queue& q,
                  ForwardIt first, Size count, const T& value ) {
  if(count <= 0)
    return sycl::event{};
  return q.parallel_for(sycl::range{static_cast<size_t>(count)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          *it = value;
                        });
}


template<class ForwardIt, class Generator >
sycl::event generate(sycl::queue& q, ForwardIt first, ForwardIt last, Generator g) {
  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          *it = g();
                        });
}

template<class ForwardIt, class Size, class Generator >
sycl::event generate_n(sycl::queue& q, ForwardIt first,
                      Size count, Generator g) {
  if(count <= 0)
    return sycl::event{};
  return q.parallel_for(sycl::range{static_cast<size_t>(count)},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          *it = g();
                        });
}

template<class ForwardIt, class T>
sycl::event replace(sycl::queue& q, ForwardIt first, ForwardIt last,
                    const T& old_value, const T& new_value) {
  return for_each(q, first, last, [=](auto& x){
    if(x == old_value)
      x = new_value;
  });
}

template<class ForwardIt,
         class UnaryPredicate, class T >
sycl::event replace_if(sycl::queue& q, ForwardIt first, ForwardIt last,
                      UnaryPredicate p, const T& new_value) {
  return for_each(q, first, last, [=](auto& x){
    if(p(x))
      x = new_value;
  });
}

template <class ForwardIt1, class ForwardIt2,
          class UnaryPredicate, class T>
sycl::event replace_copy_if(
    sycl::queue& q, ForwardIt1 first,
    ForwardIt1 last, ForwardIt2 d_first, UnaryPredicate p, const T &new_value) {
  return q.parallel_for(sycl::range{std::distance(first, last)},
                        [=](sycl::id<1> id) {
                          auto input = first;
                          auto output = d_first;
                          std::advance(input, id[0]);
                          std::advance(output, id[0]);
                          if (p(*input)) {
                            *output = new_value;
                          } else {
                            *output = *input;
                          }
                        });
}

template <class ForwardIt1, class ForwardIt2, class T>
sycl::event
replace_copy(sycl::queue& q,
             ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first,
             const T &old_value, const T &new_value) {
  return replace_copy_if(
      q, first, last, d_first, [=](const auto &x) { return x == old_value; },
      new_value);
}

/*
// Need transform_reduce functionality for find etc, so forward
// declare here.
template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt first, ForwardIt last, T* out, T init,
                 BinaryReductionOp reduce, UnaryTransformOp transform);

template <class ForwardIt, class T>
sycl::event find(sycl::queue &q, util::allocation_group &scratch_allocations, ForwardIt first, ForwardIt last,
                 typename std::iterator_traits<ForwardIt>::difference_type* out, const T &value) {
  using difference_type = typename std::iterator_traits<ForwardIt>::difference_type;
  
  return transform_reduce(q, scratch_allocations, first, last, out, std::distance(first, last), sycl::minimum<difference_type>{},)
}

template <class ForwardIt, class UnaryPredicate>
sycl::event find_if(sycl::queue &q, util::allocation_group &scratch_allocations, ForwardIt first, ForwardIt last,
                    typename std::iterator_traits<ForwardIt>::difference_type* out, UnaryPredicate p);

template <class ForwardIt, class UnaryPredicate>
sycl::event find_if_not(sycl::queue &q, util::allocation_group &scratch_allocations, ForwardIt first, ForwardIt last,
                        typename std::iterator_traits<ForwardIt>::difference_type* out, UnaryPredicate p);
*/                        
}

#endif
