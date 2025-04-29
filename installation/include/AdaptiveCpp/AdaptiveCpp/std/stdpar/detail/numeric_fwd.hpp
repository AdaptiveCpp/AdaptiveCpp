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
#ifndef HIPSYCL_PSTL_NUMERIC_FWD_HPP
#define HIPSYCL_PSTL_NUMERIC_FWD_HPP


#include "execution_fwd.hpp"
#include "stdpar_defs.hpp"
#include <iterator>

namespace std {

template<class ForwardIt1, class ForwardIt2, class T >
HIPSYCL_STDPAR_ENTRYPOINT
T transform_reduce(hipsycl::stdpar::par_unseq,
                    ForwardIt1 first1, ForwardIt1 last1,
                    ForwardIt2 first2,
                    T init);

template <class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>
HIPSYCL_STDPAR_ENTRYPOINT T transform_reduce(
    hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
    T init, BinaryReductionOp reduce, BinaryTransformOp transform);

template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
HIPSYCL_STDPAR_ENTRYPOINT T transform_reduce(hipsycl::stdpar::par_unseq,
                                             ForwardIt first, ForwardIt last,
                                             T init, BinaryReductionOp reduce,
                                             UnaryTransformOp transform);

template <class ForwardIt>
HIPSYCL_STDPAR_ENTRYPOINT typename std::iterator_traits<ForwardIt>::value_type
reduce(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last);

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT T reduce(hipsycl::stdpar::par_unseq, ForwardIt first,
                                   ForwardIt last, T init);

template <class ForwardIt, class T, class BinaryOp>
HIPSYCL_STDPAR_ENTRYPOINT T reduce(hipsycl::stdpar::par_unseq, ForwardIt first,
                                   ForwardIt last, T init, BinaryOp binary_op);
}

#endif
