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

#ifndef HIPSYCL_PSTL_ALGORITHM_DEFINITION_HPP
#define HIPSYCL_PSTL_ALGORITHM_DEFINITION_HPP

#include "hipSYCL/std/stdpar/algorithm"

#include "../detail/sycl_glue.hpp"
#include "../detail/stdpar_builtins.hpp"
#include "hipSYCL/algorithms/algorithm.hpp"

namespace std {

template<class ForwardIt, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT
void for_each(std::execution::offload_parallel_unsequenced_policy,
               ForwardIt first, ForwardIt last, UnaryFunction2 f) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  hipsycl::algorithms::for_each(q, first, last, f);
  __hipsycl_stdpar_optimizable_sync(q);
}

template<class ForwardIt, class Size, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt for_each_n(std::execution::offload_parallel_unsequenced_policy,
                    ForwardIt first, Size n, UnaryFunction2 f) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  ForwardIt last = first;
  std::advance(last, std::max(n, 0));
  hipsycl::algorithms::for_each_n(q, first, n, f);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 transform(std::execution::offload_parallel_unsequenced_policy,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first,
                     UnaryOperation unary_op) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto last = d_first;
  std::advance(last, std::distance(first1, last1));
  hipsycl::algorithms::transform(q, first1, last1, d_first, unary_op);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 transform(std::execution::offload_parallel_unsequenced_policy,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                     ForwardIt3 d_first, BinaryOperation binary_op) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto last = d_first;
  std::advance(last, std::distance(first1, last1));
  hipsycl::algorithms::transform(q, first1, last1, first2, d_first, binary_op);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy(std::execution::offload_parallel_unsequenced_policy,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto d_last = d_first;
  std::advance(d_last, std::distance(first, last));
  hipsycl::algorithms::copy(q, first, last, d_first);
  __hipsycl_stdpar_optimizable_sync(q);
  return d_last;
}

template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_if(std::execution::offload_parallel_unsequenced_policy,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first,
                   UnaryPredicate pred) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto d_last = d_first;
  std::advance(d_last, std::distance(first, last));
  hipsycl::algorithms::copy_if(q, first, last, d_first, pred);
  __hipsycl_stdpar_optimizable_sync(q);
  return d_last;
}

template<class ForwardIt1, class Size, class ForwardIt2 >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_n(std::execution::offload_parallel_unsequenced_policy,
                   ForwardIt1 first, Size count, ForwardIt2 result ) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto last = result;
  std::advance(last, std::max(count, 0));
  hipsycl::algorithms::copy_n(q, first, count, result);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template<class ForwardIt, class T >
HIPSYCL_STDPAR_ENTRYPOINT
void fill(std::execution::offload_parallel_unsequenced_policy,
          ForwardIt first, ForwardIt last, const T& value) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  hipsycl::algorithms::fill(q, first, last, value);
  __hipsycl_stdpar_optimizable_sync(q);
}

template<class ForwardIt, class Size, class T >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt fill_n(std::execution::offload_parallel_unsequenced_policy,
                  ForwardIt first, Size count, const T& value ) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto last = first;
  std::advance(last, std::max(count, 0));
  hipsycl::algorithms::fill_n(q, first, count, value);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template<class ForwardIt, class Generator >
HIPSYCL_STDPAR_ENTRYPOINT
void generate( std::execution::offload_parallel_unsequenced_policy,
               ForwardIt first, ForwardIt last, Generator g) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  hipsycl::algorithms::generate(q, first, last, g);
  __hipsycl_stdpar_optimizable_sync(q);
}

template<class ForwardIt, class Size, class Generator >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt generate_n(std::execution::offload_parallel_unsequenced_policy, 
                    ForwardIt first, Size count, Generator g) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto last = first;
  std::advance(last, std::max(count, 0));
  hipsycl::algorithms::generate_n(q, first, count, g);
  __hipsycl_stdpar_optimizable_sync(q);
  return last;
}

template <class ForwardIt, class T>
void replace(std::execution::offload_parallel_unsequenced_policy,
             ForwardIt first, ForwardIt last, const T &old_value,
             const T &new_value) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  hipsycl::algorithms::replace(q, first, last, old_value, new_value);
  __hipsycl_stdpar_optimizable_sync(q);
}

template <class ForwardIt, class UnaryPredicate, class T>
void replace_if(std::execution::offload_parallel_unsequenced_policy,
                ForwardIt first, ForwardIt last, UnaryPredicate p,
                const T &new_value) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  hipsycl::algorithms::replace_if(q, first, last, p, new_value);
  __hipsycl_stdpar_optimizable_sync(q);
}

template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2
replace_copy(std::execution::offload_parallel_unsequenced_policy,
             ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first,
             const T &old_value, const T &new_value) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto d_last = d_first;
  std::advance(d_last, std::distance(first, last));
  hipsycl::algorithms::replace_copy(q, first, last, d_first, old_value,
                                    new_value);
  __hipsycl_stdpar_optimizable_sync(q);
  return d_last;
}

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 replace_copy_if(
    std::execution::offload_parallel_unsequenced_policy, ForwardIt1 first,
    ForwardIt1 last, ForwardIt2 d_first, UnaryPredicate p, const T &new_value) {
  __hipsycl_stdpar_consume_sync();
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  auto d_last = d_first;
  std::advance(d_last, std::distance(first, last));
  hipsycl::algorithms::replace_copy_if(q, first, last, d_first, p,
                                    new_value);
  __hipsycl_stdpar_optimizable_sync(q);
  return d_last;
}

}

#endif
