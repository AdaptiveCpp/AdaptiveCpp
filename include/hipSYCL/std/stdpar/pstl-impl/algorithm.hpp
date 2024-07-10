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

namespace std {

template <class ForwardIt, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT void for_each(hipsycl::stdpar::par_unseq, ForwardIt first,
                                        ForwardIt last, UnaryFunction2 f) {
  auto offloader = [&](auto& queue) {
    hipsycl::algorithms::for_each(queue, first, last, f);
  };

  auto fallback = [&](){
    std::for_each(hipsycl::stdpar::par_unseq_host_fallback, first, last, f);
  };

  HIPSYCL_STDPAR_OFFLOAD_NORET(hipsycl::stdpar::algorithm_type::for_each{},
                               std::distance(first, last), offloader, fallback,
                               first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), f);
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

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::for_each_n{}, n,
                         ForwardIt, offloader, fallback, first, n, f);
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
      hipsycl::stdpar::algorithm_type::transform{},
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
      hipsycl::stdpar::algorithm_type::transform{},
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

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::copy{},
                         std::distance(first, last), ForwardIt2, offloader,
                         fallback, first,
                         HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first);
}

template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
HIPSYCL_STDPAR_ENTRYPOINT
ForwardIt2 copy_if(hipsycl::stdpar::par_unseq,
                   ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 d_first,
                   UnaryPredicate pred) {
  auto offloader = [&](auto& queue){
    ForwardIt2 d_last = d_first;
    std::advance(d_last, std::distance(first, last));
    hipsycl::algorithms::copy_if(queue, first, last, d_first, pred);
    return d_last;
  };

  auto fallback = [&]() {
    return std::copy_if(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                        d_first, pred);
  };

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::copy_if{},
                         std::distance(first, last), ForwardIt2, offloader,
                         fallback, first,
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

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::copy_n{}, count,
                         ForwardIt2, offloader, fallback, first, count, result);
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

  HIPSYCL_STDPAR_OFFLOAD_NORET(hipsycl::stdpar::algorithm_type::fill{},
                               std::distance(first, last), offloader, fallback,
                               first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
                               value);
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

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::fill_n{}, count,
                         ForwardIt, offloader, fallback, first, count, value);
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
      hipsycl::stdpar::algorithm_type::generate{}, std::distance(first, last),
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), g);
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

  HIPSYCL_STDPAR_OFFLOAD(hipsycl::stdpar::algorithm_type::generate_n{}, count,
                         ForwardIt, offloader, fallback, first, count, g);
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

  HIPSYCL_STDPAR_OFFLOAD_NORET(hipsycl::stdpar::algorithm_type::replace{},
                               std::distance(first, last), offloader, fallback,
                               first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last),
                               old_value, new_value);
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

  HIPSYCL_STDPAR_OFFLOAD_NORET(hipsycl::stdpar::algorithm_type::replace_if{},
                               std::distance(first, last), offloader, fallback,
                               first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p,
                               new_value);
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
      hipsycl::stdpar::algorithm_type::replace_copy{},
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
      hipsycl::stdpar::algorithm_type::replace_copy_if{},
      std::distance(first, last), ForwardIt2, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), d_first, p, new_value);
}

/*
template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find(const hipsycl::stdpar::par_unseq, ForwardIt first,
                                         ForwardIt last, const T &value);

template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if(const hipsycl::stdpar::par_unseq,
                                            ForwardIt first, ForwardIt last,
                                            UnaryPredicate p);

template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if_not(const hipsycl::stdpar::par_unseq,
                                                ForwardIt first, ForwardIt last,
                                                UnaryPredicate q); */


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
                hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::all_of(queue, first, last, output, p);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::all_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(hipsycl::stdpar::algorithm_type::all_of{},
                                  std::distance(first, last), bool, offloader,
                                  fallback, first,
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
                hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::any_of(queue, first, last, output, p);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::any_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(hipsycl::stdpar::algorithm_type::any_of{},
                                  std::distance(first, last), bool, offloader,
                                  fallback, first,
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
                hipsycl::algorithms::util::allocation_type::host>();

    auto *output = output_scratch_group
                      .obtain<hipsycl::algorithms::detail::early_exit_flag_t>(1);
    hipsycl::algorithms::none_of(queue, first, last, output, p);
    queue.wait();
    return static_cast<bool>(*output);
  };

  auto fallback = [&](){
    return std::none_of(hipsycl::stdpar::par_unseq_host_fallback, first, last, p);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(hipsycl::stdpar::algorithm_type::none_of{},
                                  std::distance(first, last), bool, offloader,
                                  fallback, first,
                                  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), p);
}

}

#endif
