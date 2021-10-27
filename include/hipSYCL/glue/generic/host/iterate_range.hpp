/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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

#ifndef HIPSYCL_ITERATE_RANGE_HPP
#define HIPSYCL_ITERATE_RANGE_HPP

#include <cstdint>

#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"

namespace hipsycl {
namespace glue {
namespace host {

template <int Dim, class Function>
void iterate_range(const sycl::range<Dim> r, Function f) noexcept {

  if constexpr (Dim == 1) {
    for (std::size_t i = 0; i < r.get(0); ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    for (std::size_t i = 0; i < r.get(0); ++i) {
      for (std::size_t j = 0; j < r.get(1); ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
    for (std::size_t i = 0; i < r.get(0); ++i) {
      for (std::size_t j = 0; j < r.get(1); ++j) {
        for (std::size_t k = 0; k < r.get(2); ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

// Iterate range by subdividing it into tiles of given size. 
// The argument passed into f is the index of the tiles.
template <int Dim, class Function>
void iterate_range_tiles(const sycl::range<Dim> r,
                         const sycl::range<Dim> tile_size,
                         Function&& f) noexcept {

  if constexpr (Dim == 1) {
    for (std::size_t i = 0; i * tile_size.get(0) < r.get(0); ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    for (std::size_t i = 0; i * tile_size.get(0) < r.get(0); ++i) {
      for (std::size_t j = 0; j * tile_size.get(1) < r.get(1); ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
    for (std::size_t i = 0; i * tile_size.get(0) < r.get(0); ++i) {
      for (std::size_t j = 0; j * tile_size.get(1) < r.get(1); ++j) {
        for (std::size_t k = 0; k * tile_size.get(2) < r.get(2); ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

template <int Dim, class Function>
void iterate_range_simd(const sycl::range<Dim> r, Function&& f) noexcept {

  if constexpr (Dim == 1) {
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < r.get(0); ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    const size_t r0 = r.get(0);
    const size_t r1 = r.get(1);

    for (std::size_t i = 0; i < r0; ++i) {
#ifdef _OPENMP
#pragma omp simd
#endif
      for (std::size_t j = 0; j < r1; ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
    const size_t r0 = r.get(0);
    const size_t r1 = r.get(1);
    const size_t r2 = r.get(2);

    for (std::size_t i = 0; i < r0; ++i) {
      for (std::size_t j = 0; j < r1; ++j) {
#ifdef _OPENMP
#pragma omp simd
#endif
        for (std::size_t k = 0; k < r2; ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

template <int Dim, class Function>
void iterate_range(const sycl::id<Dim> offset,
                   const sycl::range<Dim> r,
                   Function f) noexcept {

  const std::size_t min_i = offset.get(0);
  const std::size_t max_i = offset.get(0) + r.get(0);

  if constexpr (Dim == 1) {
    for (std::size_t i = min_i; i < max_i; ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    const std::size_t min_j = offset.get(1);
    const std::size_t max_j = offset.get(1) + r.get(1);

    for (std::size_t i = min_i; i < max_i; ++i) {
      for (std::size_t j = min_j; j < max_j; ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
    const std::size_t min_j = offset.get(1);
    const std::size_t min_k = offset.get(2);
    const std::size_t max_j = offset.get(1) + r.get(1);
    const std::size_t max_k = offset.get(2) + r.get(2);

    for (std::size_t i = min_i; i < max_i; ++i) {
      for (std::size_t j = min_j; j < max_j; ++j) {
        for (std::size_t k = min_k; k < max_k; ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

template <int Dim, class Function>
void iterate_partial_range(const sycl::range<Dim> whole_range,
                           const sycl::id<Dim> begin,
                           const std::size_t num_elements,
                           Function f) noexcept {

  if constexpr (Dim == 1) {
    for (std::size_t i = begin.get(0); i < begin.get(0) + num_elements; ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    std::size_t n = 0;

    std::size_t i = begin.get(0);
    std::size_t j = begin.get(1);
    for (; i < whole_range.get(0); ++i) {
      for (; j < whole_range.get(1); ++j) {
        if (n >= num_elements)
          return;
        
        f(sycl::id<Dim>{i, j});
        ++n;
      }
      j = 0;
    }
  } else if constexpr (Dim == 3) {
    std::size_t n = 0;

    std::size_t i = begin.get(0);
    std::size_t j = begin.get(1);
    std::size_t k = begin.get(2);
    for (; i < whole_range.get(0); ++i) {
      for (; j < whole_range.get(1); ++j) {
        for (; k < whole_range.get(2); ++k) {
          if (n >= num_elements)
            return;
          
          f(sycl::id<Dim>{i, j, k});
          ++n;
        }
        k = 0;
      }
      j = 0;
    }
  }
}
}
}
} // namespace hipsycl

#endif
