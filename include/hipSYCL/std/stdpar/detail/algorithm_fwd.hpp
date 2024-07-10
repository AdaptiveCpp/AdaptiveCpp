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
#ifndef HIPSYCL_PSTL_ALGORITHM_FWD_HPP
#define HIPSYCL_PSTL_ALGORITHM_FWD_HPP


#include "execution_fwd.hpp"
#include "stdpar_defs.hpp"

namespace std {

template <class ForwardIt, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT void for_each(hipsycl::stdpar::par_unseq, ForwardIt first,
                                        ForwardIt last, UnaryFunction2 f);

template <class ForwardIt, class Size, class UnaryFunction2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt for_each_n(hipsycl::stdpar::par_unseq,
                                               ForwardIt first, Size n,
                                               UnaryFunction2 f);

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 transform(hipsycl::stdpar::par_unseq,
                                               ForwardIt1 first1,
                                               ForwardIt1 last1,
                                               ForwardIt2 d_first,
                                               UnaryOperation unary_op);

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt3
transform(hipsycl::stdpar::par_unseq, ForwardIt1 first1, ForwardIt1 last1,
          ForwardIt2 first2, ForwardIt3 d_first, BinaryOperation binary_op);

template <class ForwardIt1, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy(hipsycl::stdpar::par_unseq,
                                          ForwardIt1 first, ForwardIt1 last,
                                          ForwardIt2 d_first);

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy_if(hipsycl::stdpar::par_unseq,
                                             ForwardIt1 first, ForwardIt1 last,
                                             ForwardIt2 d_first,
                                             UnaryPredicate pred);

template <class ForwardIt1, class Size, class ForwardIt2>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2 copy_n(hipsycl::stdpar::par_unseq,
                                            ForwardIt1 first, Size count,
                                            ForwardIt2 result);

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT void fill(hipsycl::stdpar::par_unseq, ForwardIt first,
                                    ForwardIt last, const T &value);

template <class ForwardIt, class Size, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt fill_n(hipsycl::stdpar::par_unseq, ForwardIt first,
                                           Size count, const T &value);

template <class ForwardIt, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT void generate(hipsycl::stdpar::par_unseq, ForwardIt first,
                                        ForwardIt last, Generator g);

template <class ForwardIt, class Size, class Generator>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt generate_n(const hipsycl::stdpar::par_unseq,
                                               ForwardIt first, Size count,
                                               Generator g);

template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT void replace(hipsycl::stdpar::par_unseq, ForwardIt first,
                                       ForwardIt last, const T &old_value,
                                       const T &new_value);

template <class ForwardIt, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT void replace_if(hipsycl::stdpar::par_unseq, ForwardIt first,
                                          ForwardIt last, UnaryPredicate p,
                                          const T &new_value);

template <class ForwardIt1, class ForwardIt2, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2
replace_copy(hipsycl::stdpar::par_unseq, ForwardIt1 first, ForwardIt1 last,
             ForwardIt2 d_first, const T &old_value, const T &new_value);

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt2
replace_copy_if(hipsycl::stdpar::par_unseq, ForwardIt1 first, ForwardIt1 last,
                ForwardIt2 d_first, UnaryPredicate p, const T &new_value);

/*
template <class ForwardIt, class T>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find(hipsycl::stdpar::par_unseq, ForwardIt first,
                                         ForwardIt last, const T &value);

template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if(hipsycl::stdpar::par_unseq,
                                            ForwardIt first, ForwardIt last,
                                            UnaryPredicate p);

template <class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT ForwardIt find_if_not(hipsycl::stdpar::par_unseq,
                                                ForwardIt first, ForwardIt last,
                                                UnaryPredicate q); */


template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool all_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p );

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool any_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p );

template<class ForwardIt, class UnaryPredicate>
HIPSYCL_STDPAR_ENTRYPOINT
bool none_of(hipsycl::stdpar::par_unseq, ForwardIt first, ForwardIt last,
            UnaryPredicate p );

}

#endif
