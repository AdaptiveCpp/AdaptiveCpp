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

#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "hipSYCL/sycl/libkernel/atomic_builtins.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/functional.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "util/traits.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/algorithms/util/memory_streaming.hpp"

namespace hipsycl::algorithms {

namespace detail {

template<class T>
bool all_bytes_equal(const T& val, unsigned char& byte_value) {
  std::array<unsigned char, sizeof(T)> buff;
  std::memcpy(buff.data(), &val, sizeof(T));

  for(int i = 0; i < sizeof(T); ++i) {
    if(buff[i] != buff[0])
      return false;
  }
  byte_value = buff[0];
  return true;
}

inline bool should_use_memcpy(const sycl::device& dev) {
  // OpenMP backend implements queue::memcpy() using std::memcpy
  // which can break perf on NUMA systems
  if(dev.get_backend() == sycl::backend::omp)
    return false;
  // Some OpenCL implementations (e.g. Intel GPU) seem to be very
  // inefficient for memcpy calls between two pointers if data
  // is source and dest are on the same device (does it always go
  // through the host?)
  if(dev.get_backend() == sycl::backend::ocl)
    return false;
  if(dev.get_backend() == sycl::backend::hip)
    // It was reported that hipMemcpy does not handly copies involving
    // shared allocations efficiently
    return false;
  return true;
}

inline bool should_use_memset(const sycl::device& dev) {
  if(dev.get_backend() == sycl::backend::omp)
    return false;
  if(dev.get_backend() == sycl::backend::ocl)
    return false;
  if(dev.get_backend() == sycl::backend::hip)
    return false;
  return true;
}

}

template <class ForwardIt, class UnaryFunction2>
sycl::event for_each(sycl::queue &q, ForwardIt first, ForwardIt last,
                     UnaryFunction2 f) {
  if(first == last)
    return sycl::event{};
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
  if(first1 == last1)
    return sycl::event{};
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
  if(first1 == last1)
    return sycl::event{};
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
  
  auto size = std::distance(first, last);
  if(size == 0)
    return sycl::event{};
  
  using value_type1 = typename std::iterator_traits<ForwardIt1>::value_type;
  using value_type2 = typename std::iterator_traits<ForwardIt2>::value_type;

  if (std::is_trivially_copyable_v<value_type1> &&
      std::is_same_v<value_type1, value_type2> &&
      util::is_contiguous<ForwardIt1>() && util::is_contiguous<ForwardIt2>() &&
      detail::should_use_memcpy(q.get_device())) {
    return q.memcpy(&(*d_first), &(*first), size * sizeof(value_type1));
  } else {
    return q.parallel_for(sycl::range{size},
                          [=](sycl::id<1> id) {
                            auto input = first;
                            auto output = d_first;
                            std::advance(input, id[0]);
                            std::advance(output, id[0]);
                            *output = *input;
                          });
  }
}


template<class ForwardIt1, class ForwardIt2, class UnaryPredicate >
sycl::event copy_if(sycl::queue& q,
                    ForwardIt1 first, ForwardIt1 last,
                    ForwardIt2 d_first,
                    UnaryPredicate pred) {
  if(first == last)
    return sycl::event{};
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

  auto last = first;
  std::advance(last, count);
  return copy(q, first, last, result);
}

template <class ForwardIt, class T>
sycl::event fill(sycl::queue &q, ForwardIt first, ForwardIt last,
                 const T &value) {
  auto size = std::distance(first, last);
  if(size == 0)
    return sycl::event{};

  using value_type = typename std::iterator_traits<ForwardIt>::value_type;

  auto invoke_kernel = [&]() -> sycl::event{
    return q.parallel_for(sycl::range{size},
                        [=](sycl::id<1> id) {
                          auto it = first;
                          std::advance(it, id[0]);
                          *it = value;
                        });
  };

  if constexpr (std::is_trivial_v<value_type> &&
                std::is_same_v<value_type, T> &&
                util::is_contiguous<ForwardIt>()) {
    unsigned char equal_byte;
    if (detail::all_bytes_equal(value, equal_byte) &&
        detail::should_use_memset(q.get_device())) {
      return q.memset(&(*first), static_cast<int>(equal_byte),
                      size * sizeof(T));
    } else {
      return invoke_kernel();
    }
  } else {
    return invoke_kernel();
  }
}

template<class ForwardIt, class Size, class T >
sycl::event fill_n(sycl::queue& q,
                  ForwardIt first, Size count, const T& value ) {
  if(count <= Size{0})
    return sycl::event{};
  
  auto last = first;
  std::advance(last, count);
  return fill(q, first, last, value);
}


template<class ForwardIt, class Generator >
sycl::event generate(sycl::queue& q, ForwardIt first, ForwardIt last, Generator g) {
  if(first == last)
    return sycl::event{};
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
  if(first == last)
    return sycl::event{};
  return for_each(q, first, last, [=](auto& x){
    if(x == old_value)
      x = new_value;
  });
}

template<class ForwardIt,
         class UnaryPredicate, class T >
sycl::event replace_if(sycl::queue& q, ForwardIt first, ForwardIt last,
                      UnaryPredicate p, const T& new_value) {
  if(first == last)
    return sycl::event{};
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
  if (first == last)
    return sycl::event{};
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
  if (first == last)
    return sycl::event{};
  return replace_copy_if(
      q, first, last, d_first, [=](const auto &x) { return x == old_value; },
      new_value);
}

// Need transform_reduce functionality for find etc, so forward
// declare here.
template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt first, ForwardIt last, T* out, T init,
                 BinaryReductionOp reduce, UnaryTransformOp transform);

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

namespace detail {
using early_exit_flag_t = int;

// predicate must be a callable of type bool(sycl::id<1>).
// If it returns true, the for_each will abort and output_has_exited_early
// will be set to true.
template <class Predicate>
sycl::event early_exit_for_each(sycl::queue &q, std::size_t problem_size,
                                early_exit_flag_t *output_has_exited_early,
                                Predicate should_exit) {
  
  std::size_t group_size = 128;

  util::abortable_data_streamer streamer{q.get_device(), problem_size, group_size};

  std::size_t dispatched_global_size = streamer.get_required_global_size();

  auto kernel = [=](sycl::nd_item<1> idx) {
      const std::size_t item_id = idx.get_global_id(0);
  
      util::abortable_data_streamer::run(problem_size, idx, [&](sycl::id<1> idx){
        
        if (sycl::detail::__acpp_atomic_load<
                sycl::access::address_space::global_space>(
                output_has_exited_early, sycl::memory_order_relaxed,
                sycl::memory_scope_device)) {
          return true;
        }

        if (should_exit(idx)) {
          sycl::detail::__acpp_atomic_store<
              sycl::access::address_space::global_space>(
              output_has_exited_early, 1, sycl::memory_order_relaxed,
              sycl::memory_scope_device);
          return true;
        }

        return false;
      });
    };

  auto evt = q.single_task([=](){*output_has_exited_early = false;});
  return q.parallel_for(sycl::nd_range<1>{dispatched_global_size, group_size}, evt,
                        kernel);
}

}

template <class ForwardIt, class UnaryPredicate>
sycl::event all_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, detail::early_exit_flag_t* out,
                   UnaryPredicate p) {
  std::size_t problem_size = std::distance(first, last);
  if(problem_size == 0)
    return sycl::event{};
  auto evt = detail::early_exit_for_each(q, problem_size, out,
                                     [=](sycl::id<1> idx) -> bool {
                                       auto it = first;
                                       std::advance(it, idx[0]);
                                       return !p(*it);
                                     });
  return q.single_task(evt, [=](){
    *out = static_cast<detail::early_exit_flag_t>(!(*out));
  });
}

template <class ForwardIt, class UnaryPredicate>
sycl::event any_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, detail::early_exit_flag_t* out,
                   UnaryPredicate p) {
  std::size_t problem_size = std::distance(first, last);
  if(problem_size == 0)
    return sycl::event{};
  return detail::early_exit_for_each(q, problem_size, out,
                                     [=](sycl::id<1> idx) -> bool {
                                       auto it = first;
                                       std::advance(it, idx[0]);
                                       return p(*it);
                                     });
}

template <class ForwardIt, class UnaryPredicate>
sycl::event none_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, detail::early_exit_flag_t* out,
                   UnaryPredicate p) {
  std::size_t problem_size = std::distance(first, last);
  if(problem_size == 0)
    return sycl::event{};
  
  auto evt = any_of(q, first, last, out, p);
  return q.single_task(evt, [=](){
    *out = static_cast<detail::early_exit_flag_t>(!(*out));
  });
}

}

#endif
