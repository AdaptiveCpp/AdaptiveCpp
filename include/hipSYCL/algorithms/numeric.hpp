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
#ifndef HIPSYCL_ALGORITHMS_NUMERIC_HPP
#define HIPSYCL_ALGORITHMS_NUMERIC_HPP

#include <cstddef>
#include <iterator>
#include <functional>
#include <limits>

#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "hipSYCL/sycl/libkernel/functional.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/algorithms/reduction/reduction_descriptor.hpp"
#include "hipSYCL/algorithms/reduction/reduction_engine.hpp"
#include "hipSYCL/algorithms/util/memory_streaming.hpp"

namespace hipsycl::algorithms {

namespace detail {

template<class T, class Op>
struct identity {
  static constexpr bool is_known() { return false; }
};

#define HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(op, value)                    \
  template <class T> struct identity<T, op> {                                  \
    static constexpr bool is_known() { return true; }                          \
    static T get_identity() { return value; }                                  \
  };

// TODO: Need to restrict this to fundamental types
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(std::plus<T>, T{})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(std::multiplies<T>, T{1})

HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::plus<T>, T{})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::multiplies<T>, T{1})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::minimum<T>, std::numeric_limits<T>::max())
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::maximum<T>, std::numeric_limits<T>::min())


template<class T, class BinaryOp>
auto get_reduction_operator_configuration(const BinaryOp& op) {
  if constexpr(detail::identity<T, BinaryOp>::is_known()) {
    return reduction::reduction_binary_operator<T, BinaryOp, true>{
        op, detail::identity<T, BinaryOp>::get_identity()};
  } else {
    return reduction::reduction_binary_operator<T, BinaryOp, false>{op};
  }
}


template <class T, class Kernel,
          class BinaryReductionOp>
sycl::event wg_model_reduction(sycl::queue &q,
                               util::allocation_group &scratch_allocations,
                               T *output, T init, std::size_t target_num_groups,
                               std::size_t local_size, std::size_t problem_size,
                               Kernel k, BinaryReductionOp op) {
  assert(target_num_groups > 0);

  sycl::event last_event;
  auto ndrange_launcher =
      [&](std::size_t num_groups, std::size_t wg_size, std::size_t global_size,
          std::size_t local_mem, auto kernel) {
        last_event = q.submit([&](sycl::handler &cgh) {
          // This is just there to register the appropriate amount of local
          // memory; the reduction engine will access it directly without going
          // through the accessor.
          sycl::local_accessor<char> acc{sycl::range<1>{local_mem}, cgh};
          cgh.parallel_for(sycl::nd_range<1>{wg_size * num_groups, wg_size},
                           kernel);
        });
      };

  auto operator_config = get_reduction_operator_configuration<T>(op);
  auto reduction_descriptor = reduction::reduction_descriptor{
      operator_config, init, output};

  using group_reduction_type =
      reduction::wg_model::group_reductions::generic_local_memory<
          std::decay_t<decltype(reduction_descriptor)>>;
  
  // The reduction engine will update this value with the
  // appropriate amount of local memory for the main kernel.
  std::size_t main_kernel_local_mem = 0;
  reduction::wg_model::group_horizontal_reducer<group_reduction_type>
      horizontal_reducer{
          group_reduction_type{main_kernel_local_mem, local_size}};
  reduction::wg_hierarchical_reduction_engine engine{horizontal_reducer,
                                                     &scratch_allocations};

  util::data_streamer streamer{q.get_device(), problem_size, local_size};

  const std::size_t dispatched_global_size =
      streamer.get_required_global_size();
  auto plan = engine.create_plan(dispatched_global_size, local_size,
                                reduction_descriptor);

  auto main_kernel = engine.make_main_reducing_kernel(
      [=](sycl::nd_item<1> idx, auto &reducer) {

        util::data_streamer::run(problem_size, idx, [&](sycl::id<1> i){
          k(i, reducer);
        });
      },
      plan);

  last_event = q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<char> acc{sycl::range<1>{main_kernel_local_mem}, cgh};
    cgh.parallel_for(sycl::nd_range<1>{dispatched_global_size, local_size},
                    main_kernel);
  });

  engine.run_additional_kernels(ndrange_launcher, plan);

  
  return last_event;
}

template <class T, class Kernel, class BinaryReductionOp>
sycl::event
wg_model_reduction(sycl::queue &q, util::allocation_group &scratch_allocations,
                   T *output, T init, std::size_t target_num_groups,
                   std::size_t problem_size, Kernel k, BinaryReductionOp op) {
  return wg_model_reduction(q, scratch_allocations, output, init,
                                  target_num_groups, 128, problem_size, k, op);
}

template <class T, class Kernel, class BinaryReductionOp>
sycl::event threading_model_reduction(sycl::queue &q,
                                  util::allocation_group &scratch_allocations,
                                  T *output, T init, std::size_t n, Kernel k,
                                  BinaryReductionOp op) {

  sycl::event last_event;
  auto single_task_launcher =
      [&](auto kernel) {
        last_event = q.single_task(kernel);
      };

  auto operator_config = get_reduction_operator_configuration<T>(op);
  auto reduction_descriptor = reduction::reduction_descriptor{
      operator_config, init, output};
  
  reduction::threading_model::omp_thread_info_query thread_info_query;
  reduction::threading_reduction_engine engine{thread_info_query,
                                               &scratch_allocations};
  auto plan = engine.create_plan(n, reduction_descriptor);
  auto main_kernel = engine.make_main_reducing_kernel(k, plan);
  
  last_event = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{n},
                     main_kernel);
  });

  engine.run_additional_kernels(single_task_launcher, plan);
  
  return last_event;
}

template <class T, class Kernel, class BinaryReductionOp>
sycl::event transform_reduce_impl(sycl::queue &q,
                                  util::allocation_group &scratch_allocations,
                                  T *output, T init, std::size_t n, Kernel k,
                                  BinaryReductionOp op) {
  if(q.get_device().is_host()) {
#ifdef HIPSYCL_ALGORITHMS_TRANSFORM_REDUCE_HOST_THREADING_MODEL
    return threading_model_reduction(q, scratch_allocations, output, init, n, k,
                                     op);
#endif
  }
  sycl::device dev = q.get_device();
  std::size_t num_groups =
      dev.get_info<sycl::info::device::max_compute_units>() * 4;

  return wg_model_reduction(q, scratch_allocations, output, init, num_groups,
                            n, k, op);

}

}

// Note: All transform_reduce variants defined here behave slightly different than STL
// variants:
// * If first==last, returns an event that is complete, even if preceding enqueued operations
//   are not yet complete.
// * If first==last, *out remains untouched and will not contain the init value.
template <class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T *out,
                 T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform) {
  if(first1 == last1)
    return sycl::event{};
  
  std::size_t n = std::distance(first1, last1);
  auto kernel = [=](sycl::id<1> idx, auto& reducer) {
    auto input_a = first1;
    auto input_b = first2;
    std::advance(input_a, idx[0]);
    std::advance(input_b, idx[0]);
    reducer.combine(transform(*input_a, *input_b));
  };

  return detail::transform_reduce_impl(q, scratch_allocations, out, init, n,
                                       kernel, reduce);
}

template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt first, ForwardIt last, T* out, T init,
                 BinaryReductionOp reduce, UnaryTransformOp transform) {
  if(first == last)
    return sycl::event{};
  
  std::size_t n = std::distance(first, last);
  auto kernel = [=](sycl::id<1> idx, auto& reducer) {
    auto input = first;
    std::advance(input, idx[0]);
    reducer.combine(transform(*input));
  };

  return detail::transform_reduce_impl(q, scratch_allocations, out, init, n,
                                       kernel, reduce);
}

template <class ForwardIt1, class ForwardIt2, class T>
sycl::event transform_reduce(sycl::queue &q,
                             util::allocation_group &scratch_allocations,
                             ForwardIt1 first1, ForwardIt1 last1,
                             ForwardIt2 first2, T *out, T init) {
  return transform_reduce(q, scratch_allocations, first1, last1, first2, out,
                          init, std::plus<T>{}, std::multiplies<T>{});
}


template <class ForwardIt, class T, class BinaryOp>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init,
                   BinaryOp binary_op) {
  return transform_reduce(q, scratch_allocations, first, last, out, init,
                          binary_op, [](auto x) { return x; });
}

template <class ForwardIt, class T>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init) {
  return reduce(q, scratch_allocations, first, last, out, init, std::plus<T>{});
}

template <class ForwardIt>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last,
                   typename std::iterator_traits<ForwardIt>::value_type *out) {
  return reduce(q, scratch_allocations, first, last, out,
                typename std::iterator_traits<ForwardIt>::value_type{});
}

}

#endif
