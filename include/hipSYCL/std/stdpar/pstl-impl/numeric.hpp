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

#ifndef HIPSYCL_PSTL_NUMERIC_DEFINITION_HPP
#define HIPSYCL_PSTL_NUMERIC_DEFINITION_HPP

#include "hipSYCL/std/stdpar/detail/execution_fwd.hpp"
#include "hipSYCL/std/stdpar/numeric"

#include "../detail/sycl_glue.hpp"
#include "../detail/stdpar_builtins.hpp"
#include "../detail/offload.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/algorithms/numeric.hpp"
#include <iterator>
#include <numeric>

namespace std {

template<class ForwardIt1, class ForwardIt2, class T >
HIPSYCL_STDPAR_ENTRYPOINT
T transform_reduce(hipsycl::stdpar::par_unseq,
                    ForwardIt1 first1, ForwardIt1 last1,
                    ForwardIt2 first2,
                    T init) {
  
  auto offloader = [&](auto& queue) {
    // Note: Using a scratch allocation_group that expires at the end of the scope
    // is safe because
    // a) We synchronize before the end, so the allocation_group also lives until
    // the kernels are complete;
    // b) We have one allocation cache per thread-local in-order queue. So, subsequent operations
    // fed from the same cache would wait for us anyway due to using the same in-order queue.
    // These conditions ensure that no other operations can get access to the 
    // cached scratch memory while we are using it.
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    
    T* output = output_scratch_group.obtain<T>(1);
    hipsycl::algorithms::transform_reduce(queue, reduction_scratch_group, first1,
                                            last1, first2, output, init);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();
    
    if(first1 == last1)
      return init;
    else
      return *output;
  };

  auto fallback = [&]() {
    return std::transform_reduce(hipsycl::stdpar::par_unseq_host_fallback,
                                 first1, last1, first2, init);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm_type::transform_reduce{},
      std::distance(first1, last1), T, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, init);
}

template<class ForwardIt1, class ForwardIt2, class T,
          class BinaryReductionOp,
          class BinaryTransformOp >
HIPSYCL_STDPAR_ENTRYPOINT
T transform_reduce(hipsycl::stdpar::par_unseq,
                    ForwardIt1 first1, ForwardIt1 last1,
                    ForwardIt2 first2,
                    T init,
                    BinaryReductionOp reduce,
                    BinaryTransformOp transform ) {
  auto offloader = [&](auto& queue){
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    
    T* output = output_scratch_group.obtain<T>(1);
    hipsycl::algorithms::transform_reduce(queue, reduction_scratch_group, first1,
                                          last1, first2, output, init, reduce,
                                          transform);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();
    
    if(first1 == last1)
      return init;
    else
      return *output;
  };

  auto fallback = [&]() {
    return std::transform_reduce(hipsycl::stdpar::par_unseq_host_fallback,
                                 first1, last1, first2, init, reduce,
                                 transform);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm_type::transform_reduce{},
      std::distance(first1, last1), T, offloader, fallback, first1,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last1), first2, init, reduce, transform);
}

template<class ForwardIt, class T,
          class BinaryReductionOp,
          class UnaryTransformOp >
HIPSYCL_STDPAR_ENTRYPOINT
T transform_reduce(hipsycl::stdpar::par_unseq,
                    ForwardIt first, ForwardIt last,
                    T init,
                    BinaryReductionOp reduce,
                    UnaryTransformOp transform ) {

  auto offloader = [&](auto& queue) {
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    
    T* output = output_scratch_group.obtain<T>(1);
    hipsycl::algorithms::transform_reduce(queue, reduction_scratch_group, first, last,
                                          output, init, reduce, transform);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();
    
    if(first == last)
      return init;
    else
      return *output;
  };

  auto fallback = [&]() {
    return std::transform_reduce(hipsycl::stdpar::par_unseq_host_fallback,
                                 first, last, init, reduce, transform);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm_type::transform_reduce{},
      std::distance(first, last), T, offloader, fallback, first,
      HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), init, reduce, transform);
}

template <class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT
typename std::iterator_traits<ForwardIt>::value_type
reduce(hipsycl::stdpar::par_unseq, ForwardIt first,
       ForwardIt last) {

  using result_type = typename std::iterator_traits<ForwardIt>::value_type;

  auto offloader = [&](auto &queue) {

    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();

    result_type *output = output_scratch_group.obtain<result_type>(1);
    hipsycl::algorithms::reduce(queue, reduction_scratch_group, first, last,
                                output);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();

    if (first == last)
      return result_type{};
    else
      return *output;
  };

  auto fallback = [&](){
    return std::reduce(hipsycl::stdpar::par_unseq_host_fallback, first, last);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(hipsycl::stdpar::algorithm_type::reduce{},
                                  std::distance(first, last), result_type,
                                  offloader, fallback, first,
                                  HIPSYCL_STDPAR_NO_PTR_VALIDATION(last));
}

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT
T reduce(hipsycl::stdpar::par_unseq, ForwardIt first,
         ForwardIt last, T init) {

  auto offloader = [&](auto& queue){
      
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    
    T* output = output_scratch_group.obtain<T>(1);
    hipsycl::algorithms::reduce(queue, reduction_scratch_group, first, last,
                                output, init);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();
    
    if(first == last)
      return init;
    else
      return *output;

  };

  auto fallback = [&]() {
    return std::reduce(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                       init);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm_type::reduce{}, std::distance(first, last), T,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), init);
}

template <class ForwardIt, class T, class BinaryOp>
HIPSYCL_STDPAR_ENTRYPOINT
T reduce(hipsycl::stdpar::par_unseq, ForwardIt first,
         ForwardIt last, T init, BinaryOp binary_op) {

  auto offloader = [&](auto& queue){
      
    auto output_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::host>();
    auto reduction_scratch_group =
        hipsycl::stdpar::detail::stdpar_tls_runtime::get()
            .make_scratch_group<
                hipsycl::algorithms::util::allocation_type::device>();
    
    T* output = output_scratch_group.obtain<T>(1);
    hipsycl::algorithms::reduce(queue, reduction_scratch_group, first, last, output,
                                init, binary_op);
    // We need to wait in any case here, so cannot elide synchronization
    queue.wait();
    
    if(first == last)
      return init;
    else
      return *output;
  };

  auto fallback = [&]() {
    return std::reduce(hipsycl::stdpar::par_unseq_host_fallback, first, last,
                       init, binary_op);
  };

  HIPSYCL_STDPAR_BLOCKING_OFFLOAD(
      hipsycl::stdpar::algorithm_type::reduce{}, std::distance(first, last), T,
      offloader, fallback, first, HIPSYCL_STDPAR_NO_PTR_VALIDATION(last), init,
      binary_op);
}

}

#endif
