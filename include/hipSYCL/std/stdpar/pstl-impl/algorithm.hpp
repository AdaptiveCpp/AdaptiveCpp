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
#ifndef HIPSYCL_PSTL_ALGORITHM_DEFINITION_HPP
#define HIPSYCL_PSTL_ALGORITHM_DEFINITION_HPP

#include <algorithm>
#include <iterator>

#include "../detail/execution_fwd.hpp"
#include "../detail/sycl_glue.hpp"
#include "../detail/stdpar_builtins.hpp"
#include "../detail/stdpar_defs.hpp"
#include "../detail/offload.hpp"
#include "hipSYCL/algorithms/algorithm.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/std/stdpar/detail/offload_heuristic_db.hpp"

namespace std {


////////////////// par_unseq policy

template <class ForwardIt, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT void for_each(hipsycl::stdpar::par_unseq, ForwardIt first,
                                        ForwardIt last, UnaryFunction2 f) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::for_each(queue, first, last, f);
  };

  auto fallback = [&](){
    std::for_each(hipsycl::stdpar::par_unseq_host_fallback, first, last, f);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::for_each{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), f);
}

template<class ForwardIt, class Size, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt for_each_n(hipsycl::stdpar::par_unseq,
                    ForwardIt first, Size n, UnaryFunction2 f) {
  auto offloader = [&](auto& queue) {
    ForwardIt last = first;
    std::advance(last, std::max(n, Size{0}));
    hipsycl::algorithms::for_each_n(queue, first, n, f);
    return last;
  };

  auto fallback = [&]() {
    return std::for_each_n(hipsycl::stdpar::par_unseq_host_fallback, first, n,
                           f);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::for_each_n{},
          hipsycl::stdpar::par_unseq{}),
      n, ForwardIt, offloader, fallback, first, n, f);
}

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 transform(hipsycl::stdpar::par_unseq,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first,
                     UnaryOperation unary_op) {
  
  auto offloader = [&](auto& queue){
    ForwardIt2 last = d_first;
    std::advance(last, std::distance(first1, last1));
    hipsycl::algorithms::transform(queue, first1, last1, d_first, unary_op);
    return last;
  };

  auto fallback = [&]() {
    return std::transform(hipsycl::stdpar::par_unseq_host_fallback, first1,
                          last1, d_first, unary_op);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::transform{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), ForwardIt2, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), d_first, unary_op);
}

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 transform(hipsycl::stdpar::par_unseq,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                     ForwardIt3 d_first, BinaryOperation binary_op) {

  auto offloader = [&](auto &queue) {
    ForwardIt3 last = d_first;
    std::advance(last, std::distance(first1, last1));
    hipsycl::algorithms::transform(queue, first1, last1, first2, d_first,
                                   binary_op);
    return last;
  };

  auto fallback = [&]() {
    return std::transform(hipsycl::stdpar::par_unseq_host_fallback, first1,
                          last1, first2, d_first, binary_op);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::transform{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), ForwardIt3, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, d_first, binary_op);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy(const hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first) {
  auto offloader = [&](auto& queue){
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::copy(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::copy(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_if(hipsycl::stdpar::par_unseq,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first,
                   UnaryPredicate pred) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);
    
    hipsycl::algorithms::copy_if(queue, device_scratch_group, first, last,
                                 d_first, pred, num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::copy_if(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                        d_first, pred);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy_if{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, pred);
}

template<class ForwardIt1, class Size, class ForwardIt2 >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_n(hipsycl::stdpar::par_unseq,
                   ForwardIt1 first, Size count, ForwardIt2 result ) {

  auto offloader = [&](auto& queue){
    ForwardIt2 last = result;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::copy_n(queue, first, count, result);
    return last;
  };

  auto fallback = [&]() {
    return std::copy_n(hipsycl::stdpar::par_unseq_host_fallback, first, count,
                       result);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy_n{},
                                 hipsycl::stdpar::par_unseq{}),
      count, ForwardIt2, offloader, fallback, first, count, result);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 move(hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first) {
  auto offloader = [&](auto& queue){
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::move(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::move(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::move{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template<class ForwardIt, class T >
HIPSYCL_STDPAR_ENTRYPOINT
void fill(hipsycl::stdpar::par_unseq,
          ForwardIt first, ForwardIt last, const T& value) {
  auto offloader = [&](auto& queue){
    hipsycl::algorithms::fill(queue, first, last, value);
  };

  auto fallback = [&]() {
    std::fill(hipsycl::stdpar::par_unseq_host_fallback, first, last, value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::fill{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}

template <class ForwardIt, class Size, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt fill_n(hipsycl::stdpar::par_unseq, ForwardIt first,
                                           Size count, const T &value) {
 
  auto offloader = [&](auto& queue){
    ForwardIt last = first;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::fill_n(queue, first, count, value);
    return last;
  };

  auto fallback = [&]() {
    return std::fill_n(hipsycl::stdpar::par_unseq_host_fallback, first, count,
                       value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::fill_n{},
                                 hipsycl::stdpar::par_unseq{}),
      count, ForwardIt, offloader, fallback, first, count, value);
}

template <class ForwardIt, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT void generate(hipsycl::stdpar::par_unseq, ForwardIt first,
                                        ForwardIt last, Generator g) {
  auto offloader = [&](auto &queue) {
    hipsycl::algorithms::generate(queue, first, last, g);
  };

  auto fallback = [&]() {
    std::generate(hipsycl::stdpar::par_unseq_host_fallback, first, last, g);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::generate{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), g);
}

template <class ForwardIt, class Size, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt generate_n(hipsycl::stdpar::par_unseq,
                                               ForwardIt first, Size count,
                                               Generator g) {
  auto offloader = [&](auto& queue){
    ForwardIt last = first;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::generate_n(queue, first, count, g);
    return last;
  };

  auto fallback = [&]() {
    return std::generate_n(hipsycl::stdpar::par_unseq_host_fallback, first,
                           count, g);
  };

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm(
                             hipsycl::stdpar::algorithm_category::generate_n{},
                             hipsycl::stdpar::par_unseq{}),
                         count, ForwardIt, offloader, fallback, first, count,
                         g);
}

template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 remove_copy(hipsycl::stdpar::par_unseq,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first, const T &value) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_copy(queue, device_scratch_group,
                                     first, last, d_first, value,
                                     num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_copy(hipsycl::stdpar::par_unseq_host_fallback,
                            first, last, d_first, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_copy{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, value);
}


template <class ForwardIt1, class ForwardIt2, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 remove_copy_if(hipsycl::stdpar::par_unseq, ForwardIt1 first,
                          ForwardIt1 last, ForwardIt2 d_first,
                          UnaryPredicate p) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_copy_if(queue, device_scratch_group,
                                     first, last, d_first, p,
                                     num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_copy_if(hipsycl::stdpar::par_unseq_host_fallback,
                            first, last, d_first, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_copy_if{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, p);
}


template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt remove(hipsycl::stdpar::par_unseq, ForwardIt first,
                  ForwardIt last, const T &value) {
  auto offloader = [&](auto& queue){
    if(std::distance(first, last) == 0)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove(queue, device_scratch_group, first, last,
                                value, num_elements_copied);

    queue.wait();

    auto d_last = std::next(first, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove(hipsycl::stdpar::par_unseq_host_fallback,
                            first, last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}


template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt remove_if(hipsycl::stdpar::par_unseq, ForwardIt first,
                  ForwardIt last, UnaryPredicate pred) {
  auto offloader = [&](auto& queue){
    if(std::distance(first, last) == 0)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_if(queue, device_scratch_group, first, last,
                                pred, num_elements_copied);

    queue.wait();

    auto d_last = std::next(first, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_if(hipsycl::stdpar::par_unseq_host_fallback,
                            first, last, pred);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_if{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), pred);
}


template <class ForwardIt, class T>
void replace(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
             const T &old_value, const T &new_value) {
  auto offloader = [&](auto &queue) {
    hipsycl::algorithms::replace(queue, first, last, old_value, new_value);
  };

  auto fallback = [&]() {
    std::replace(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                 old_value, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::replace{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), old_value, new_value);
}

template <class ForwardIt, class UnaryPredicate, class T>
void replace_if(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
                UnaryPredicate p, const T &new_value) {
  
  auto offloader = [&](auto& queue){
    hipsycl::algorithms::replace_if(queue, first, last, p, new_value);
  };

  auto fallback = [&]() {
    std::replace_if(hipsycl::stdpar::par_unseq_host_fallback, first, last, p,
                    new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::replace_if{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p, new_value);
}

template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2
replace_copy(hipsycl::stdpar::par_unseq, ForwardIt1 first, ForwardIt1 last,
             ForwardIt2 d_first, const T &old_value, const T &new_value) {

  auto offloader = [&](auto &queue) {
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::replace_copy(queue, first, last, d_first, old_value,
                                      new_value);
    return d_last;
  };

  auto fallback = [&]() {
    return std::replace_copy(hipsycl::stdpar::par_unseq_host_fallback, first,
                             last, d_first, old_value, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::replace_copy{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, old_value, new_value);
}

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 replace_copy_if(
    hipsycl::stdpar::par_unseq, ForwardIt1 first,
    ForwardIt1 last, ForwardIt2 d_first, UnaryPredicate p, const T &new_value) {

  auto offloader = [&](auto &queue) {
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::replace_copy_if(queue, first, last, d_first, p,
                                         new_value);
    return d_last;
  };

  auto fallback = [&]() {
    return std::replace_copy_if(hipsycl::stdpar::par_unseq_host_fallback, first,
                                last, d_first, p, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
                             hipsycl::stdpar::algorithm_category::replace_copy_if{},
                             hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, p, new_value);
}

template<class BidirIt>
HIPSYCL_STDPAR_ENTRYPOINT void reverse (hipsycl::stdpar::par_unseq,
                                        BidirIt first, BidirIt last) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::reverse(queue, first, last);
  };

  auto fallback = [&]() {
    std::reverse(hipsycl::stdpar::par_unseq_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
    hipsycl::stdpar::algorithm(
                          hipsycl::stdpar::algorithm_category::reverse{},
                          hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}

template<class BidirIt, class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt reverse_copy (hipsycl::stdpar::par_unseq,
                                                  BidirIt first, BidirIt last,
                                                  ForwardIt d_first) {
  auto offloader = [&](auto& queue) {
    ForwardIt d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::reverse_copy(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::reverse_copy(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::reverse_copy{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find(const hipsycl::stdpar::par_unseq, ForwardIt first,
                                         ForwardIt last, const T &value) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find(queue, reduction_scratch_group, first,
                              last, value, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find(hipsycl::stdpar::par_unseq_host_fallback, first,
                     last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}

template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if(const hipsycl::stdpar::par_unseq,
                                            ForwardIt first, ForwardIt last,
                                            UnaryPredicate p) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_if(queue, reduction_scratch_group, first,
                              last, p, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find_if(hipsycl::stdpar::par_unseq_host_fallback, first,
                     last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_if{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if_not(const hipsycl::stdpar::par_unseq,
                                                ForwardIt first, ForwardIt last,
                                                UnaryPredicate p) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_if_not(queue, reduction_scratch_group, first,
                              last, p, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find_if_not(hipsycl::stdpar::par_unseq_host_fallback, first,
                     last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_if_not{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_end(hipsycl::stdpar::par_unseq, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

  if (std::distance(first, last) < std::distance(s_first, s_last))
    return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_end(queue, reduction_scratch_group, first,
                              last, s_first, s_last, out);

    queue.wait();

    ForwardIt1 found_at = first;
    if (*out != std::numeric_limits<DiffT>::min()) {
      std::advance(found_at, *out);
      return found_at;
    }

    return last;
  };

  auto fallback = [&]() {
    return std::find_end(hipsycl::stdpar::par_unseq_host_fallback,
                              first, last, s_first, s_last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_end{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), ForwardIt1, offloader, fallback,
    first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), s_first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_end(hipsycl::stdpar::par_unseq, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last, BinaryPredicate p) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    if (std::distance(first, last) < std::distance(s_first, s_last))
    return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_end(queue, reduction_scratch_group, first,
                              last, s_first, s_last, p, out);

    queue.wait();

    ForwardIt1 found_at = first;
    if (*out != std::numeric_limits<DiffT>::min()) {
      std::advance(found_at, *out);
      return found_at;
    }

    return last;
  };

  auto fallback = [&]() {
    return std::find_end(hipsycl::stdpar::par_unseq_host_fallback,
                              first, last, s_first, s_last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_end{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), ForwardIt1, offloader, fallback,
    first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), s_first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_first_of(hipsycl::stdpar::par_unseq, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_first_of(queue, reduction_scratch_group, first,
                              last, s_first, s_last, out);

    queue.wait();


    ForwardIt1 found_at = first;
    std::advance(found_at, *out);
    return found_at;
  };

  auto fallback = [&]() {
    return std::find_first_of(hipsycl::stdpar::par_unseq_host_fallback,
                              first, last, s_first, s_last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(
      hipsycl::stdpar::algorithm_category::find_first_of{},
      hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), ForwardIt1, offloader,
    fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
    s_first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_first_of(hipsycl::stdpar::par_unseq, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last, BinaryPredicate p) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_first_of(queue, reduction_scratch_group, first,
                              last, s_first, s_last, p, out);

    queue.wait();


    ForwardIt1 found_at = first;
    std::advance(found_at, *out);
    return found_at;
  };

  auto fallback = [&]() {
    return std::find_first_of(hipsycl::stdpar::par_unseq_host_fallback,
                              first, last, s_first, s_last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(
      hipsycl::stdpar::algorithm_category::find_first_of{},
      hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), ForwardIt1, offloader,
    fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
    s_first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool all_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {

  auto offloader = [&](auto& queue){
    
    if(std::distance(first, last) == 0)
      return true;
    
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::all_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::all_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::all_of{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool any_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {
  
  auto offloader = [&](auto& queue){

    if(std::distance(first, last) == 0)
      return false;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::any_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::any_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::any_of{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool none_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {
  
  auto offloader = [&](auto& queue){

    if(std::distance(first, last) == 0)
      return true;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::none_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::none_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::none_of{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2) {
  auto offloader = [&](auto& queue){

      if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_unseq_host_fallback, first1, last1, first2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2);
}

template <class ForwardIt1, class ForwardIt2, class BinaryPred>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, BinaryPred p) {
  auto offloader = [&](auto& queue){

      if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, p, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_unseq_host_fallback, first1, last1, first2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, p);
}

template<class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT typename std::iterator_traits<ForwardIt>::difference_type
count(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
         const T& value) {
  auto offloader = [&](auto& queue) {
  using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;

    if(std::distance(first, last) == 0)
      return DiffT{};

    auto  output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::count(queue, reduction_scratch_group, first,
                               last, out, value);
    queue.wait();
    return *out;
  };

  auto fallback = [&]() {
    return std::count(hipsycl::stdpar::par_unseq_host_fallback, first,
                         last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::count{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), typename std::iterator_traits<ForwardIt>::difference_type,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}


template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT typename std::iterator_traits<ForwardIt>::difference_type
count_if(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
         UnaryPredicate p) {
  auto offloader = [&](auto& queue) {
  using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;

    if(std::distance(first, last) == 0)
      return DiffT{};

    auto  output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::count_if(queue, reduction_scratch_group, first,
                                  last, out, p);
    queue.wait();
    return *out;
  };

  auto fallback = [&]() {
    return std::count_if(hipsycl::stdpar::par_unseq_host_fallback, first,
                         last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::count_if{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), typename std::iterator_traits<ForwardIt>::difference_type,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2) {

  auto offloader = [&](auto& queue) {
    if(std::distance(first1, last1) == 0)
      return std::make_pair(first1, first2);

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_unseq_host_fallback,
                         first1, last1, first2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first1, last1), std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2);
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par_unseq,
                                    ForwardIt1 first1, ForwardIt1 last1,
                                    ForwardIt2 first2, BinaryPredicate p) {

  auto offloader = [&](auto& queue) {
    if(std::distance(first1, last1) == 0)
      return std::make_pair(first1, first2);

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, p, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_unseq_host_fallback,
                         first1, last1, first2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first1, last1), std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2, ForwardIt2 last2) {
    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT problem_size = std::min(std::distance(first1, last1),
                                  std::distance(first2, last2));

  auto offloader = [&](auto& queue) {
    if(first1 == last1 || first2 == last2)
      return std::make_pair(first1, first2);

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, last2, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_unseq_host_fallback,
                         first1, last1, first2, last2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par_unseq{}),
    problem_size, std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2, ForwardIt2 last2,
                                          BinaryPredicate p) {
    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT problem_size = std::min(std::distance(first1, last1),
                                  std::distance(first2, last2));

  auto offloader = [&](auto& queue) {
    if(first1 == last1 || first2 == last2)
      return std::make_pair(first1, first2);

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, last2, p, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_unseq_host_fallback,
                         first1, last1, first2, last2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par_unseq{}),
    problem_size, std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), p);
}


template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2) {
  auto offloader = [&](auto& queue){

      if (std::distance(first1, last1) != std::distance(first2, last2))
        return false;
      else if (std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, last2, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_unseq_host_fallback, first1, last1, first2, last2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, last2);
}

template <class ForwardIt1, class ForwardIt2, class BinaryPred>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2, BinaryPred p) {
  auto offloader = [&](auto& queue){

      if (std::distance(first1, last1) != std::distance(first2, last2))
        return false;
      else if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, last2, p, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_unseq_host_fallback, first1, last1,
                      first2, last2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, last2, p);
}


template <class RandomIt>
HIPSYCL_STDPAR_ENTRYPOINT void sort(hipsycl::stdpar::par_unseq, RandomIt first,
                                        RandomIt last) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::sort(queue, first, last);
  };

  auto fallback = [&](){
    std::sort(hipsycl::stdpar::par_unseq_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::sort{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template <class RandomIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT void sort(hipsycl::stdpar::par_unseq, RandomIt first,
                                        RandomIt last, Compare comp) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::sort(queue, first, last, comp);
  };

  auto fallback = [&]() {
    std::sort(hipsycl::stdpar::par_unseq_host_fallback, first, last, comp);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::sort{},
          hipsycl::stdpar::par_unseq{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT bool is_sorted(hipsycl::stdpar::par_unseq, ForwardIt first,
                                         ForwardIt last) {
  auto offloader = [&](auto &queue){
    if(first == last || std::distance(first, last) == 1)
      return true;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::is_sorted(queue, first, last, output);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&]() {
    return std::is_sorted(hipsycl::stdpar::par_unseq_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), bool, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT bool is_sorted(hipsycl::stdpar::par_unseq, ForwardIt first,
                                         ForwardIt last, Compare comp) {
  auto offloader = [&](auto &queue){
    if(first == last || std::distance(first, last) == 1)
      return true;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::is_sorted(queue, first, last, output, comp);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&]() {
    return std::is_sorted(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                          comp);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted{},
                               hipsycl::stdpar::par_unseq{}),
    std::distance(first, last), bool, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt is_sorted_until(hipsycl::stdpar::par_unseq, ForwardIt first,
                          ForwardIt last) {
  auto offloader = [&](auto &queue){
    if (first == last || std::distance(first, last) == 1)
      return last;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::is_sorted_until(queue, reduction_scratch_group, first,
                                         last, out);

    queue.wait();

    if (*out == std::distance(first, last))
      return last;

    ForwardIt sorted_until = std::next(first, *out + 1);
    return sorted_until;
  };

  auto fallback = [&]() {
    return std::is_sorted_until(hipsycl::stdpar::par_unseq_host_fallback,
                                first, last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted_until{},
                               hipsycl::stdpar::par_unseq()),
    std::distance(first, last), ForwardIt, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt is_sorted_until(hipsycl::stdpar::par_unseq, ForwardIt first,
                          ForwardIt last, Compare comp) {
  auto offloader = [&](auto &queue){
    if (first == last || std::distance(first, last) == 1)
      return last;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::is_sorted_until(queue, reduction_scratch_group, first,
                                         last, out, comp);

    queue.wait();

    if (*out == std::distance(first, last))
      return last;

    ForwardIt sorted_until = std::next(first, *out + 1);
    return sorted_until;
  };

  auto fallback = [&]() {
    return std::is_sorted_until(hipsycl::stdpar::par_unseq_host_fallback,
                                first, last, comp);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted_until{},
                               hipsycl::stdpar::par_unseq()),
    std::distance(first, last), ForwardIt, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt1, class ForwardIt2,
         class ForwardIt3, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 merge(hipsycl::stdpar::par_unseq,
                  ForwardIt1 first1, ForwardIt1 last1,
                  ForwardIt2 first2, ForwardIt2 last2,
                  ForwardIt3 d_first, Compare comp) {
  auto offloader = [&](auto &queue) {
    auto scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    hipsycl::algorithms::merge(queue, scratch_group, first1, last1, first2,
                               last2, d_first, comp);
    auto d_last = d_first;
    std::advance(d_last,
                 std::distance(first1, last1) + std::distance(first2, last2));
    return d_last;
  };

  auto fallback = [&]() {
    return std::merge(hipsycl::stdpar::par_unseq_host_fallback, first1, last1,
                      first2, last2, d_first, comp);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::merge{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1) + std::distance(first2, last2), ForwardIt3,
      offloader, fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1),
      first2, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), d_first, comp);
}

template<class ForwardIt1, class ForwardIt2,
         class ForwardIt3, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 merge(hipsycl::stdpar::par_unseq,
                  ForwardIt1 first1, ForwardIt1 last1,
                  ForwardIt2 first2, ForwardIt2 last2,
                  ForwardIt3 d_first) {
  auto offloader = [&](auto &queue) {
    auto scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    hipsycl::algorithms::merge(queue, scratch_group, first1, last1, first2,
                               last2, d_first);
    auto d_last = d_first;
    std::advance(d_last,
                 std::distance(first1, last1) + std::distance(first2, last2));
    return d_last;
  };

  auto fallback = [&]() {
    return std::merge(hipsycl::stdpar::par_unseq_host_fallback, first1, last1,
                      first2, last2, d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::merge{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1) + std::distance(first2, last2), ForwardIt3,
      offloader, fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1),
      first2, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), d_first);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt min_element(hipsycl::stdpar::par_unseq, ForwardIt first,
                      ForwardIt last) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MinPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MinPair *out = output_scratch_group.obtain<MinPair>(1);
  hipsycl::algorithms::min_element(queue, reduction_scratch_group,
                                   first, last, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::min_element(hipsycl::stdpar::par_unseq_host_fallback, first,
                          last);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::min_element{},
                             hipsycl::stdpar::par_unseq{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt min_element(hipsycl::stdpar::par_unseq, ForwardIt first,
                      ForwardIt last, Compare comp) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MinPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MinPair *out = output_scratch_group.obtain<MinPair>(1);
  hipsycl::algorithms::min_element(queue, reduction_scratch_group,
                                   first, last, comp, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::min_element(hipsycl::stdpar::par_unseq_host_fallback, first,
                          last, comp);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::min_element{},
                             hipsycl::stdpar::par_unseq{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt max_element(hipsycl::stdpar::par_unseq, ForwardIt first,
                      ForwardIt last) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MaxPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MaxPair *out = output_scratch_group.obtain<MaxPair>(1);
  hipsycl::algorithms::max_element(queue, reduction_scratch_group,
                                   first, last, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));
  return found_it;
};

auto fallback = [&]() {
  return std::max_element(hipsycl::stdpar::par_unseq_host_fallback, first,
                          last);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::max_element{},
                             hipsycl::stdpar::par_unseq{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt max_element(hipsycl::stdpar::par_unseq, ForwardIt first,
                      ForwardIt last, Compare comp) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MaxPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MaxPair *out = output_scratch_group.obtain<MaxPair>(1);
  hipsycl::algorithms::max_element(queue, reduction_scratch_group,
                                   first, last, comp, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));
  return found_it;
};

auto fallback = [&]() {
  return std::max_element(hipsycl::stdpar::par_unseq_host_fallback, first,
                          last, comp);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::max_element{},
                             hipsycl::stdpar::par_unseq{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


//////////////////// par policy  /////////////////////////////////////


template <class ForwardIt, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT void for_each(hipsycl::stdpar::par, ForwardIt first,
                                        ForwardIt last, UnaryFunction2 f) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::for_each(queue, first, last, f);
  };

  auto fallback = [&](){
    std::for_each(hipsycl::stdpar::par_host_fallback, first, last, f);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::for_each{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), f);
}

template<class ForwardIt, class Size, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt for_each_n(hipsycl::stdpar::par,
                    ForwardIt first, Size n, UnaryFunction2 f) {
  auto offloader = [&](auto& queue) {
    ForwardIt last = first;
    std::advance(last, std::max(n, Size{0}));
    hipsycl::algorithms::for_each_n(queue, first, n, f);
    return last;
  };

  auto fallback = [&]() {
    return std::for_each_n(hipsycl::stdpar::par_host_fallback, first, n,
                           f);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::for_each_n{},
          hipsycl::stdpar::par{}),
      n, ForwardIt, offloader, fallback, first, n, f);
}

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 transform(hipsycl::stdpar::par,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first,
                     UnaryOperation unary_op) {
  
  auto offloader = [&](auto& queue){
    ForwardIt2 last = d_first;
    std::advance(last, std::distance(first1, last1));
    hipsycl::algorithms::transform(queue, first1, last1, d_first, unary_op);
    return last;
  };

  auto fallback = [&]() {
    return std::transform(hipsycl::stdpar::par_host_fallback, first1,
                          last1, d_first, unary_op);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::transform{},
          hipsycl::stdpar::par{}),
      std::distance(first1, last1), ForwardIt2, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), d_first, unary_op);
}

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 transform(hipsycl::stdpar::par,
                     ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                     ForwardIt3 d_first, BinaryOperation binary_op) {

  auto offloader = [&](auto &queue) {
    ForwardIt3 last = d_first;
    std::advance(last, std::distance(first1, last1));
    hipsycl::algorithms::transform(queue, first1, last1, first2, d_first,
                                   binary_op);
    return last;
  };

  auto fallback = [&]() {
    return std::transform(hipsycl::stdpar::par_host_fallback, first1,
                          last1, first2, d_first, binary_op);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::transform{},
          hipsycl::stdpar::par{}),
      std::distance(first1, last1), ForwardIt3, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, d_first, binary_op);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy(const hipsycl::stdpar::par,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first) {
  auto offloader = [&](auto& queue){
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::copy(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::copy(hipsycl::stdpar::par_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_if(hipsycl::stdpar::par,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first,
                   UnaryPredicate pred) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);
    
    hipsycl::algorithms::copy_if(queue, device_scratch_group, first, last,
                                 d_first, pred, num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::copy_if(hipsycl::stdpar::par_host_fallback, first, last,
                        d_first, pred);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy_if{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, pred);
}

template<class ForwardIt1, class Size, class ForwardIt2 >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_n(hipsycl::stdpar::par,
                   ForwardIt1 first, Size count, ForwardIt2 result ) {

  auto offloader = [&](auto& queue){
    ForwardIt2 last = result;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::copy_n(queue, first, count, result);
    return last;
  };

  auto fallback = [&]() {
    return std::copy_n(hipsycl::stdpar::par_host_fallback, first, count,
                       result);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::copy_n{},
                                 hipsycl::stdpar::par{}),
      count, ForwardIt2, offloader, fallback, first, count, result);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 move(const hipsycl::stdpar::par,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first) {
  auto offloader = [&](auto& queue){
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::move(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::move(hipsycl::stdpar::par_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::move{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template<class ForwardIt, class T >
HIPSYCL_STDPAR_ENTRYPOINT
void fill(hipsycl::stdpar::par,
          ForwardIt first, ForwardIt last, const T& value) {
  auto offloader = [&](auto& queue){
    hipsycl::algorithms::fill(queue, first, last, value);
  };

  auto fallback = [&]() {
    std::fill(hipsycl::stdpar::par_host_fallback, first, last, value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::fill{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}

template <class ForwardIt, class Size, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt fill_n(hipsycl::stdpar::par, ForwardIt first,
                                           Size count, const T &value) {
 
  auto offloader = [&](auto& queue){
    ForwardIt last = first;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::fill_n(queue, first, count, value);
    return last;
  };

  auto fallback = [&]() {
    return std::fill_n(hipsycl::stdpar::par_host_fallback, first, count,
                       value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::fill_n{},
                                 hipsycl::stdpar::par{}),
      count, ForwardIt, offloader, fallback, first, count, value);
}

template <class ForwardIt, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT void generate(hipsycl::stdpar::par, ForwardIt first,
                                        ForwardIt last, Generator g) {
  auto offloader = [&](auto &queue) {
    hipsycl::algorithms::generate(queue, first, last, g);
  };

  auto fallback = [&]() {
    std::generate(hipsycl::stdpar::par_host_fallback, first, last, g);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::generate{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), g);
}

template <class ForwardIt, class Size, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt generate_n(hipsycl::stdpar::par,
                                               ForwardIt first, Size count,
                                               Generator g) {
  auto offloader = [&](auto& queue){
    ForwardIt last = first;
    std::advance(last, std::max(count, Size{0}));
    hipsycl::algorithms::generate_n(queue, first, count, g);
    return last;
  };

  auto fallback = [&]() {
    return std::generate_n(hipsycl::stdpar::par_host_fallback, first,
                           count, g);
  };

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm(
                             hipsycl::stdpar::algorithm_category::generate_n{},
                             hipsycl::stdpar::par{}),
                         count, ForwardIt, offloader, fallback, first, count,
                         g);
}


template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 remove_copy(hipsycl::stdpar::par,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first, const T &value) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_copy(queue, device_scratch_group,
                                     first, last, d_first, value,
                                     num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_copy(hipsycl::stdpar::par_host_fallback,
                            first, last, d_first, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_copy{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, value);
}

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 remove_copy_if(hipsycl::stdpar::par, ForwardIt1 first,
                          ForwardIt1 last, ForwardIt2 d_first,
                          UnaryPredicate p) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_copy_if(queue, device_scratch_group,
                                     first, last, d_first, p,
                                     num_elements_copied);
    queue.wait();

    ForwardIt2 d_last = d_first;
    std::advance(d_last, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_copy_if(hipsycl::stdpar::par_host_fallback,
                            first, last, d_first, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_copy_if{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, p);
}


template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt remove(hipsycl::stdpar::par, ForwardIt first,
                  ForwardIt last, const T &value) {
  auto offloader = [&](auto& queue){
    if(std::distance(first, last) == 0)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove(queue, device_scratch_group, first, last,
                                value, num_elements_copied);

    queue.wait();

    auto d_last = std::next(first, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove(hipsycl::stdpar::par_host_fallback,
                            first, last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}


template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt remove_if(hipsycl::stdpar::par, ForwardIt first,
                  ForwardIt last, UnaryPredicate pred) {
  auto offloader = [&](auto& queue){
    if(std::distance(first, last) == 0)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto device_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    std::size_t *num_elements_copied =
        output_scratch_group.obtain<std::size_t>(1);

    hipsycl::algorithms::remove_if(queue, device_scratch_group, first, last,
                                pred, num_elements_copied);

    queue.wait();

    auto d_last = std::next(first, *num_elements_copied);
    return d_last;
  };

  auto fallback = [&]() {
    return std::remove_if(hipsycl::stdpar::par_host_fallback,
                            first, last, pred);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::remove_if{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), pred);
}


template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT
void replace(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
             const T &old_value, const T &new_value) {
  auto offloader = [&](auto &queue) {
    hipsycl::algorithms::replace(queue, first, last, old_value, new_value);
  };

  auto fallback = [&]() {
    std::replace(hipsycl::stdpar::par_host_fallback, first, last,
                 old_value, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::replace{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), old_value, new_value);
}

template <class ForwardIt, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT
void replace_if(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
                UnaryPredicate p, const T &new_value) {
  
  auto offloader = [&](auto& queue){
    hipsycl::algorithms::replace_if(queue, first, last, p, new_value);
  };

  auto fallback = [&]() {
    std::replace_if(hipsycl::stdpar::par_host_fallback, first, last, p,
                    new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::replace_if{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p, new_value);
}

template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2
replace_copy(hipsycl::stdpar::par, ForwardIt1 first, ForwardIt1 last,
             ForwardIt2 d_first, const T &old_value, const T &new_value) {

  auto offloader = [&](auto &queue) {
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::replace_copy(queue, first, last, d_first, old_value,
                                      new_value);
    return d_last;
  };

  auto fallback = [&]() {
    return std::replace_copy(hipsycl::stdpar::par_host_fallback, first,
                             last, d_first, old_value, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::replace_copy{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, old_value, new_value);
}

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 replace_copy_if(
    hipsycl::stdpar::par, ForwardIt1 first,
    ForwardIt1 last, ForwardIt2 d_first, UnaryPredicate p, const T &new_value) {

  auto offloader = [&](auto &queue) {
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::replace_copy_if(queue, first, last, d_first, p,
                                         new_value);
    return d_last;
  };

  auto fallback = [&]() {
    return std::replace_copy_if(hipsycl::stdpar::par_host_fallback, first,
                                last, d_first, p, new_value);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(
                             hipsycl::stdpar::algorithm_category::replace_copy_if{},
                             hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, p, new_value);
}

template<class BidirIt>
HIPSYCL_STDPAR_ENTRYPOINT void reverse (hipsycl::stdpar::par,
                                        BidirIt first, BidirIt last) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::reverse(queue, first, last);
  };

  auto fallback = [&]() {
    std::reverse(hipsycl::stdpar::par_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
    hipsycl::stdpar::algorithm(
                          hipsycl::stdpar::algorithm_category::reverse{},
                          hipsycl::stdpar::par{}),
    std::distance(first, last), offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}

template<class BidirIt, class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt reverse_copy (hipsycl::stdpar::par,
                                                  BidirIt first, BidirIt last,
                                                  ForwardIt d_first) {
  auto offloader = [&](auto& queue) {
    ForwardIt d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::reverse_copy(queue, first, last, d_first);
    return d_last;
  };

  auto fallback = [&]() {
    return std::reverse_copy(hipsycl::stdpar::par_host_fallback, first, last,
                     d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::reverse_copy{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find(const hipsycl::stdpar::par, ForwardIt first,
                                         ForwardIt last, const T &value) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find(queue, reduction_scratch_group, first,
                              last, value, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find(hipsycl::stdpar::par_host_fallback, first,
                     last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}


template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if(const hipsycl::stdpar::par,
                                            ForwardIt first, ForwardIt last,
                                            UnaryPredicate p) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_if(queue, reduction_scratch_group, first,
                              last, p, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find_if(hipsycl::stdpar::par_host_fallback, first,
                     last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_if{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if_not(const hipsycl::stdpar::par,
                                                ForwardIt first, ForwardIt last,
                                                UnaryPredicate p) {
  auto offloader = [&](auto& queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_if_not(queue, reduction_scratch_group, first,
                              last, p, out);

    queue.wait();

    if(first == last)
      return last;
    else {
      ForwardIt found_at = first;
      std::advance(found_at, *out);
      return found_at;
    }
  };

  auto fallback =[&]() {
    return std::find_if_not(hipsycl::stdpar::par_host_fallback, first,
                     last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_if_not{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), ForwardIt, offloader, fallback,
      first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_end(hipsycl::stdpar::par, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

  if (std::distance(first, last) < std::distance(s_first, s_last))
    return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_end(queue, reduction_scratch_group, first,
                              last, s_first, s_last, out);

    queue.wait();

    ForwardIt1 found_at = first;
    if (*out != std::numeric_limits<DiffT>::min()) {
      std::advance(found_at, *out);
      return found_at;
    }

    return last;
  };

  auto fallback = [&]() {
    return std::find_end(hipsycl::stdpar::par_host_fallback,
                              first, last, s_first, s_last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_end{},
                               hipsycl::stdpar::par{}),
    std::distance(first, last), ForwardIt1, offloader, fallback,
    first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), s_first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_end(hipsycl::stdpar::par, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last, BinaryPredicate p) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    if (std::distance(first, last) < std::distance(s_first, s_last))
    return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_end(queue, reduction_scratch_group, first,
                              last, s_first, s_last, p, out);

    queue.wait();

    ForwardIt1 found_at = first;
    if (*out != std::numeric_limits<DiffT>::min()) {
      std::advance(found_at, *out);
      return found_at;
    }

    return last;
  };

  auto fallback = [&]() {
    return std::find_end(hipsycl::stdpar::par_host_fallback,
                              first, last, s_first, s_last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::find_end{},
                               hipsycl::stdpar::par{}),
    std::distance(first, last), ForwardIt1, offloader, fallback,
    first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), s_first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_first_of(hipsycl::stdpar::par, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_first_of(queue, reduction_scratch_group, first,
                              last, s_first, s_last, out);

    queue.wait();

    ForwardIt1 found_at = first;
    std::advance(found_at, *out);
    return found_at;
  };

  auto fallback = [&]() {
    return std::find_first_of(hipsycl::stdpar::par_host_fallback,
                              first, last, s_first, s_last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(
      hipsycl::stdpar::algorithm_category::find_first_of{},
      hipsycl::stdpar::par{}),
    std::distance(first, last), ForwardIt1, offloader,
    fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
    s_first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt1 find_first_of(hipsycl::stdpar::par, ForwardIt1 first,
                         ForwardIt1 last, ForwardIt2 s_first,
                         ForwardIt2 s_last, BinaryPredicate p) {
  auto offloader = [&](auto &queue) {
    if(first == last || s_first == s_last)
      return last;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::find_first_of(queue, reduction_scratch_group, first,
                              last, s_first, s_last, p, out);

    queue.wait();

    ForwardIt1 found_at = first;
    std::advance(found_at, *out);
    return found_at;
  };

  auto fallback = [&]() {
    return std::find_first_of(hipsycl::stdpar::par_host_fallback,
                              first, last, s_first, s_last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(
      hipsycl::stdpar::algorithm_category::find_first_of{},
      hipsycl::stdpar::par{}),
    std::distance(first, last), ForwardIt1, offloader,
    fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
    s_first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(s_last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool all_of(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {

  auto offloader = [&](auto& queue){
    
    if(std::distance(first, last) == 0)
      return true;
    
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::all_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::all_of(hipsycl::stdpar::par_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::all_of{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool any_of(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {
  
  auto offloader = [&](auto& queue){

    if(std::distance(first, last) == 0)
      return false;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::any_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::any_of(hipsycl::stdpar::par_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::any_of{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool none_of(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
            UnaryPredicate p ) {
  
  auto offloader = [&](auto& queue){

    if(std::distance(first, last) == 0)
      return true;

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::none_of(queue, first, last, output, p);
    hipsycl::algorithms::detail::early_exit_flag_t result;
    queue.memcpy(&result, output, sizeof(hipsycl::algorithms::detail::early_exit_flag_t));
    queue.wait();
    return static_cast<bool>(result);
  };

  auto fallback = [&](){
    return std::none_of(hipsycl::stdpar::par_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::none_of{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), bool, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template<class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT typename std::iterator_traits<ForwardIt>::difference_type
count(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
         const T& value) {
  auto offloader = [&](auto& queue) {
  using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;

    if(std::distance(first, last) == 0)
      return DiffT{};

    auto  output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::count(queue, reduction_scratch_group, first,
                               last, out, value);
    queue.wait();
    return *out;
  };

  auto fallback = [&]() {
    return std::count(hipsycl::stdpar::par_host_fallback, first,
                         last, value);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::count{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), typename std::iterator_traits<ForwardIt>::difference_type,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), value);
}


template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT typename std::iterator_traits<ForwardIt>::difference_type
count_if(hipsycl::stdpar::par, ForwardIt first, ForwardIt last,
         UnaryPredicate p) {
  auto offloader = [&](auto& queue) {
  using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;

    if(std::distance(first, last) == 0)
      return DiffT{};

    auto  output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::count_if(queue, reduction_scratch_group, first,
                                  last, out, p);
    queue.wait();
    return *out;
  };

  auto fallback = [&]() {
    return std::count_if(hipsycl::stdpar::par_host_fallback, first,
                         last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::count_if{},
                                 hipsycl::stdpar::par{}),
      std::distance(first, last), typename std::iterator_traits<ForwardIt>::difference_type,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2) {
  auto offloader = [&](auto& queue) {
    if(std::distance(first1, last1) == 0)
      return std::make_pair(first1, first2);

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_host_fallback,
                         first1, last1, first2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par{}),
    std::distance(first1, last1), std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2);
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par,
                                    ForwardIt1 first1, ForwardIt1 last1,
                                    ForwardIt2 first2, BinaryPredicate p) {

  auto offloader = [&](auto& queue) {
    if(std::distance(first1, last1) == 0)
      return std::make_pair(first1, first2);

    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, p, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_host_fallback,
                         first1, last1, first2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par{}),
    std::distance(first1, last1), std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, p);
}


template<class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2, ForwardIt2 last2) {
    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT problem_size = std::min(std::distance(first1, last1),
                                  std::distance(first2, last2));

  auto offloader = [&](auto& queue) {
    if(first1 == last1 || first2 == last2)
      return std::make_pair(first1, first2);

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, last2, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_host_fallback,
                         first1, last1, first2, last2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par{}),
    problem_size, std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2));
}


template<class ForwardIt1, class ForwardIt2, class BinaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
std::pair<ForwardIt1, ForwardIt2> mismatch(hipsycl::stdpar::par,
                                          ForwardIt1 first1, ForwardIt1 last1,
                                          ForwardIt2 first2, ForwardIt2 last2,
                                          BinaryPredicate p) {
    using DiffT = typename std::iterator_traits<ForwardIt1>::difference_type;
    DiffT problem_size = std::min(std::distance(first1, last1),
                                  std::distance(first2, last2));

  auto offloader = [&](auto& queue) {
    if(first1 == last1 || first2 == last2)
      return std::make_pair(first1, first2);

    auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    auto *output = output_scratch_group.obtain<DiffT>(1);

    hipsycl::algorithms::mismatch(queue, reduction_scratch_group, first1,
                                  last1, first2, last2, p, output);

    queue.wait();

    auto input1 = std::next(first1, *output);
    auto input2 = std::next(first2, *output);
    return std::make_pair(input1, input2);
  };

  auto fallback = [&]() {
    return std::mismatch(hipsycl::stdpar::par_host_fallback,
                         first1, last1, first2, last2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::mismatch{},
                               hipsycl::stdpar::par{}),
    problem_size, std::pair, offloader,
    fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), p);
}


template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2) {
  auto offloader = [&](auto& queue){

      if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_host_fallback, first1, last1, first2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2);
}


template <class ForwardIt1, class ForwardIt2, class BinaryPred>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, BinaryPred p) {
  auto offloader = [&](auto& queue){

      if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, p, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_host_fallback, first1, last1, first2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, p);
}

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2) {
  auto offloader = [&](auto& queue){

      if (std::distance(first1, last1) != std::distance(first2, last2))
        return false;
      else if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, last2, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_host_fallback, first1, last1, first2, last2);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2));
}

template <class ForwardIt1, class ForwardIt2, class BinaryPred>
HIPSYCL_STDPAR_ENTRYPOINT
bool equal(hipsycl::stdpar::par, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2, BinaryPred p) {
  auto offloader = [&](auto& queue){

      if (std::distance(first1, last1) != std::distance(first2, last2))
        return false;
      else if(std::distance(first1, last1) == 0)
        return true;

      auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

      auto *output = output_scratch_group
                        .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
      hipsycl::algorithms::equal(queue, first1, last1, first2, last2, p, output);
      queue.wait();
      return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::equal(hipsycl::stdpar::par_host_fallback, first1, last1,
                      first2, last2, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::equal{},
                                 hipsycl::stdpar::par{}),
      std::distance(first1, last1), bool, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), p);
}


template <class RandomIt>
HIPSYCL_STDPAR_ENTRYPOINT void sort(hipsycl::stdpar::par, RandomIt first,
                                        RandomIt last) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::sort(queue, first, last);
  };

  auto fallback = [&](){
    std::sort(hipsycl::stdpar::par_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::sort{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}

template <class RandomIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT void sort(hipsycl::stdpar::par, RandomIt first,
                                    RandomIt last, Compare comp) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::sort(queue, first, last, comp);
  };

  auto fallback = [&]() {
    std::sort(hipsycl::stdpar::par_host_fallback, first, last, comp);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(
      hipsycl::stdpar::algorithm(
          hipsycl::stdpar::algorithm_category::sort{},
          hipsycl::stdpar::par{}),
      std::distance(first, last), offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}

template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT bool is_sorted(hipsycl::stdpar::par, ForwardIt first,
                                         ForwardIt last) {
  auto offloader = [&](auto &queue){
    if(first == last || std::distance(first, last) == 1)
      return true;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::is_sorted(queue, first, last, output);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&]() {
    return std::is_sorted(hipsycl::stdpar::par_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted{},
                               hipsycl::stdpar::par{}),
    std::distance(first, last), bool, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT bool is_sorted(hipsycl::stdpar::par, ForwardIt first,
                                         ForwardIt last, Compare comp) {
  auto offloader = [&](auto &queue){
    if(first == last || std::distance(first, last) == 1)
      return true;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::is_sorted(queue, first, last, output, comp);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&]() {
    return std::is_sorted(hipsycl::stdpar::par_host_fallback, first, last,
                          comp);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted{},
                               hipsycl::stdpar::par{}),
    std::distance(first, last), bool, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt is_sorted_until(hipsycl::stdpar::par, ForwardIt first,
                          ForwardIt last) {
  auto offloader = [&](auto &queue){
    if (first == last || std::distance(first, last) == 1)
      return last;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::is_sorted_until(queue, reduction_scratch_group, first,
                                         last, out);

    queue.wait();

    if (*out == std::distance(first, last))
      return last;

    ForwardIt sorted_until = std::next(first, *out + 1);
    return sorted_until;
  };

  auto fallback = [&]() {
    return std::is_sorted_until(hipsycl::stdpar::par_host_fallback,
                                first, last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted_until{},
                               hipsycl::stdpar::par()),
    std::distance(first, last), ForwardIt, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt is_sorted_until(hipsycl::stdpar::par, ForwardIt first,
                          ForwardIt last, Compare comp) {
  auto offloader = [&](auto &queue){
    if (first == last || std::distance(first, last) == 1)
      return last;

    auto output_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::host>();

    auto reduction_scratch_group =
          hipsycl::stdpar::detail::stdpar_tls_runtime::get()
              .make_scratch_group<
                  hipsycl::algorithms::util::allocation_type::device>();

    using DiffT = typename std::iterator_traits<ForwardIt>::difference_type;
    DiffT *out = output_scratch_group.obtain<DiffT>(1);
    hipsycl::algorithms::is_sorted_until(queue, reduction_scratch_group, first,
                                         last, out, comp);

    queue.wait();

    if (*out == std::distance(first, last))
      return last;

    ForwardIt sorted_until = std::next(first, *out + 1);
    return sorted_until;
  };

  auto fallback = [&]() {
    return std::is_sorted_until(hipsycl::stdpar::par_host_fallback,
                                first, last, comp);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
    hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::is_sorted_until{},
                               hipsycl::stdpar::par()),
    std::distance(first, last), ForwardIt, offloader, fallback, first,
    HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt1, class ForwardIt2,
         class ForwardIt3, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 merge(hipsycl::stdpar::par,
                  ForwardIt1 first1, ForwardIt1 last1,
                  ForwardIt2 first2, ForwardIt2 last2,
                  ForwardIt3 d_first, Compare comp) {
  auto offloader = [&](auto &queue) {
    auto scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    hipsycl::algorithms::merge(queue, scratch_group, first1, last1, first2,
                               last2, d_first, comp);
    auto d_last = d_first;
    std::advance(d_last,
                 std::distance(first1, last1) + std::distance(first2, last2));
    return d_last;
  };

  auto fallback = [&]() {
    return std::merge(hipsycl::stdpar::par_unseq_host_fallback, first1, last1,
                      first2, last2, d_first, comp);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::merge{},
                                 hipsycl::stdpar::par_unseq{}),
      std::distance(first1, last1) + std::distance(first2, last2), ForwardIt3,
      offloader, fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1),
      first2, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), d_first, comp);
}

template<class ForwardIt1, class ForwardIt2,
         class ForwardIt3, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt3 merge(hipsycl::stdpar::par,
                  ForwardIt1 first1, ForwardIt1 last1,
                  ForwardIt2 first2, ForwardIt2 last2,
                  ForwardIt3 d_first) {
  auto offloader = [&](auto &queue) {
    auto scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    hipsycl::algorithms::merge(queue, scratch_group, first1, last1, first2,
                               last2, d_first);
    auto d_last = d_first;
    std::advance(d_last,
                 std::distance(first1, last1) + std::distance(first2, last2));
    return d_last;
  };

  auto fallback = [&]() {
    return std::merge(hipsycl::stdpar::par_host_fallback, first1, last1,
                      first2, last2, d_first);
  };

  HIPSYCL_STDPAR_OFFLOAD(
      hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::merge{},
                                 hipsycl::stdpar::par{}),
      std::distance(first1, last1) + std::distance(first2, last2), ForwardIt3,
      offloader, fallback, first1, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1),
      first2, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last2), d_first);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt min_element(hipsycl::stdpar::par, ForwardIt first,
                      ForwardIt last) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MinPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MinPair *out = output_scratch_group.obtain<MinPair>(1);
  hipsycl::algorithms::min_element(queue, reduction_scratch_group,
                                   first, last, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::min_element(hipsycl::stdpar::par_host_fallback, first,
                          last);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::min_element{},
                             hipsycl::stdpar::par{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt min_element(hipsycl::stdpar::par, ForwardIt first,
                      ForwardIt last, Compare comp) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MinPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MinPair *out = output_scratch_group.obtain<MinPair>(1);
  hipsycl::algorithms::min_element(queue, reduction_scratch_group,
                                   first, last, comp, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::min_element(hipsycl::stdpar::par_host_fallback, first,
                          last, comp);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::min_element{},
                             hipsycl::stdpar::par{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


template<class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt max_element(hipsycl::stdpar::par, ForwardIt first,
                      ForwardIt last) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MaxPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MaxPair *out = output_scratch_group.obtain<MaxPair>(1);
  hipsycl::algorithms::max_element(queue, reduction_scratch_group,
                                   first, last, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::max_element(hipsycl::stdpar::par_host_fallback, first,
                          last);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::max_element{},
                             hipsycl::stdpar::par{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}


template<class ForwardIt, class Compare>
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt max_element(hipsycl::stdpar::par, ForwardIt first,
                      ForwardIt last, Compare comp) {
auto offloader = [&](auto &queue) {
  if (first == last)
    return last;

  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  using MaxPair = std::pair<ForwardIt, ValueT>;

  auto output_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::host>();

  auto reduction_scratch_group =
      hipsycl::stdpar::detail::stdpar_tls_runtime::get()
          .make_scratch_group<
              hipsycl::algorithms::util::allocation_type::device>();

  MaxPair *out = output_scratch_group.obtain<MaxPair>(1);
  hipsycl::algorithms::max_element(queue, reduction_scratch_group,
                                   first, last, comp, out);

  queue.wait();

  ForwardIt found_it = first;
  std::advance(found_it, std::distance(first, (*out).first));

  return found_it;
};

auto fallback = [&]() {
  return std::max_element(hipsycl::stdpar::par_host_fallback, first,
                          last, comp);
};

HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
  hipsycl::stdpar::algorithm(hipsycl::stdpar::algorithm_category::max_element{},
                             hipsycl::stdpar::par{}),
  std::distance(first, last), ForwardIt, offloader, fallback, first,
  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), comp);
}


}

#endif
