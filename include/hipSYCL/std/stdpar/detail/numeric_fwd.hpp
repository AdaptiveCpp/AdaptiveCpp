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
