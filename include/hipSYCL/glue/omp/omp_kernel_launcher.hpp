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

#ifndef HIPSYCL_OPENMP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_OPENMP_KERNEL_LAUNCHER_HPP

#include <cassert>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/libkernel/detail/data_layout.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

#include "../generic/host/collective_execution_engine.hpp"
#include "../generic/host/iterate_range.hpp"
#include "../generic/host/sequential_reducer.hpp"

namespace hipsycl {
namespace glue {
namespace omp_dispatch {

inline int get_my_thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline int get_max_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int get_num_threads() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

template <class ReductionDescriptor> class omp_reducer {
public:
  omp_reducer(host::sequential_reducer<ReductionDescriptor>& seq_reducer)
      : _seq_reducer{seq_reducer}, _my_thread_id{get_my_thread_id()} {}

  using value_type =
      typename host::sequential_reducer<ReductionDescriptor>::value_type;
  using combiner_type =
      typename host::sequential_reducer<ReductionDescriptor>::combiner_type;

  value_type identity() const { return _seq_reducer.identity(); }
  void combine(const value_type &v) {
    _seq_reducer.combine(_my_thread_id, v);
  }

private:
  host::sequential_reducer<ReductionDescriptor>& _seq_reducer;
  int _my_thread_id;
};

template <class SequentialReducer>
void finalize_reduction(SequentialReducer &reducer) {
  reducer.finalize_result();
}

template <class Function, typename... Reductions>
void reducible_parallel_invocation(Function kernel,
                                   Reductions... reductions) noexcept {
  int max_threads = get_max_num_threads();

  auto sequential_reducers =
      std::make_tuple(host::sequential_reducer{max_threads, reductions}...);
#ifdef _OPENMP
#pragma omp parallel shared(sequential_reducers)
#endif
  {
    auto make_omp_reducers = [&](auto &... seq_reducers) {
      return std::make_tuple(omp_reducer{seq_reducers}...);
    };
    auto omp_reducers = std::apply(make_omp_reducers, sequential_reducers);

    auto make_sycl_reducers = [&](auto &... omp_reds) {
      return std::make_tuple(sycl::reducer{omp_reds}...);
    };
    auto sycl_reducers = std::apply(make_sycl_reducers, omp_reducers);

    std::apply(kernel, sycl_reducers);
  }

  auto finalize_all = [&](auto &... seq_reducers) {
    (finalize_reduction(seq_reducers), ...);
  };
  std::apply(finalize_all, sequential_reducers);
}

template <int Dim, class Function>
void iterate_range_omp_for(sycl::range<Dim> r, Function f) noexcept {

  if constexpr (Dim == 1) {
#ifdef _OPENMP
    #pragma omp for
#endif
    for (std::size_t i = 0; i < r.get(0); ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
#ifdef _OPENMP
    #pragma omp for collapse(2)
#endif
    for (std::size_t i = 0; i < r.get(0); ++i) {
      for (std::size_t j = 0; j < r.get(1); ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
#ifdef _OPENMP
    #pragma omp for collapse(3)
#endif
    for (std::size_t i = 0; i < r.get(0); ++i) {
      for (std::size_t j = 0; j < r.get(1); ++j) {
        for (std::size_t k = 0; k < r.get(2); ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

template <int Dim, class Function>
void iterate_range_omp_for(sycl::id<Dim> offset, sycl::range<Dim> r,
                           Function f) noexcept {

  const std::size_t min_i = offset.get(0);
  const std::size_t max_i = offset.get(0) + r.get(0);

  if constexpr (Dim == 1) {
#ifdef _OPENMP
  #pragma omp for
#endif
    for (std::size_t i = min_i; i < max_i; ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    const std::size_t min_j = offset.get(1);
    const std::size_t max_j = offset.get(1) + r.get(1);
#ifdef _OPENMP
  #pragma omp for collapse(2)
#endif
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
#ifdef _OPENMP
  #pragma omp for collapse(3)
#endif
    for (std::size_t i = min_i; i < max_i; ++i) {
      for (std::size_t j = min_j; j < max_j; ++j) {
        for (std::size_t k = min_k; k < max_k; ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

#ifdef __HIPSYCL_USE_ACCELERATED_CPU__
extern "C" size_t __hipsycl_local_id_x;
extern "C" size_t __hipsycl_local_id_y;
extern "C" size_t __hipsycl_local_id_z;

template <int Dim, class Function, class ...Reducers>
HIPSYCL_LOOP_SPLIT_ND_KERNEL __attribute__((noinline))
inline void iterate_nd_range_omp(Function f, const sycl::id<Dim> &&group_id, const sycl::range<Dim> num_groups,
  HIPSYCL_LOOP_SPLIT_ND_KERNEL_LOCAL_SIZE_ARG const sycl::range<Dim> local_size, const sycl::id<Dim> offset,
  size_t num_local_mem_bytes, void* group_shared_memory_ptr,
  std::function<void()> &barrier_impl,
  Reducers& ... reducers) noexcept {
  if constexpr (Dim == 1) {
    sycl::id<Dim> local_id{__hipsycl_local_id_x};
    sycl::nd_item<Dim> this_item{&offset,    group_id,   local_id,
      local_size, num_groups, &barrier_impl, group_shared_memory_ptr};
    f(this_item, reducers...);
  } else if constexpr (Dim == 2) {
    sycl::id<Dim> local_id{__hipsycl_local_id_x, __hipsycl_local_id_y};
    sycl::nd_item<Dim> this_item{&offset, group_id,
      local_id, local_size, num_groups,
      &barrier_impl, group_shared_memory_ptr};
    f(this_item, reducers...);
  } else if constexpr (Dim == 3) {
    sycl::id<Dim> local_id{__hipsycl_local_id_x, __hipsycl_local_id_y, __hipsycl_local_id_z};
    sycl::nd_item<Dim> this_item{&offset,    group_id,
      local_id,   local_size,
      num_groups, &barrier_impl, group_shared_memory_ptr};
    f(this_item, reducers...);
  }
}
#endif

template<class Function>
inline
void single_task_kernel(Function f) noexcept
{
  f();
}

template <int Dim, class Function, typename... Reductions>
inline void parallel_for_kernel(Function f,
                                const sycl::range<Dim> execution_range,
                                Reductions... reductions) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

  reducible_parallel_invocation([&, f](auto& ... reducers){
    iterate_range_omp_for(execution_range, [&](sycl::id<Dim> idx) {
      auto this_item =
        sycl::detail::make_item<Dim>(idx, execution_range);

      f(this_item, reducers...);
    });
  }, reductions...);
}

template <int Dim, class Function, typename... Reductions>
inline void parallel_for_kernel_offset(Function f,
                                       const sycl::range<Dim> execution_range,
                                       const sycl::id<Dim> offset,
                                       Reductions... reductions) noexcept {
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

  reducible_parallel_invocation([&, f](auto& ... reducers){
    iterate_range_omp_for(offset, execution_range, [&](sycl::id<Dim> idx) {
      auto this_item =
        sycl::detail::make_item<Dim>(idx, execution_range, offset);

      f(this_item, reducers...);
    });
  }, reductions...);
}

template <int Dim, class Function, typename... Reductions>
inline void parallel_for_ndrange_kernel(
    Function f, const sycl::range<Dim> num_groups,
    const sycl::range<Dim> local_size, const sycl::id<Dim> offset,
    size_t num_local_mem_bytes, Reductions... reductions) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1 - 3 are supported.");

  reducible_parallel_invocation([&, f](auto& ... reducers){
    if(num_groups.size() == 0 || local_size.size() == 0)
      return;

    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    // 128 kiB as local memory for group algorithms
    std::aligned_storage_t<128*1024, sizeof(double) * 16> group_shared_memory_ptr{};
#ifdef __HIPSYCL_USE_ACCELERATED_CPU__
    std::function<void()> barrier_impl = [] () noexcept {
      assert(false && "splitting seems to have failed");
      std::terminate();
    };

    iterate_range_omp_for(num_groups, [&](sycl::id<Dim> &&group_id) {
      iterate_nd_range_omp(f, std::move(group_id), num_groups, local_size, offset,
        num_local_mem_bytes, &group_shared_memory_ptr, barrier_impl, reducers...);
    });
#elif defined(HIPSYCL_HAS_FIBERS)
    host::static_range_decomposition<Dim> group_decomposition{
        num_groups, get_num_threads()};

    host::collective_execution_engine<Dim> engine{num_groups, local_size,
                                                  offset, group_decomposition,
                                                  get_my_thread_id()};

    std::function<void()> barrier_impl = [&]() { engine.barrier(); };

    engine.run_kernel([&](sycl::id<Dim> local_id, sycl::id<Dim> group_id) {

      auto linear_group_id =
          sycl::detail::linear_id<Dim>::get(group_id, num_groups);

      sycl::nd_item<Dim> this_item{&offset,
                                    group_id,
                                    local_id,
                                    local_size,
                                    num_groups,
                                    &barrier_impl,
                                    &group_shared_memory_ptr};

      f(this_item, reducers...);
    });
#endif

    sycl::detail::host_local_memory::release();

  }, reductions...);
}

template <int Dim, class Function, typename... Reductions>
inline void parallel_for_workgroup(Function f,
                                   const sycl::range<Dim> num_groups,
                                   const sycl::range<Dim> local_size,
                                   size_t num_local_mem_bytes,
                                   Reductions... reductions) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

  reducible_parallel_invocation(
      [&, f, num_groups, local_size](auto &... reducers) {

        sycl::detail::host_local_memory::request_from_threadprivate_pool(
            num_local_mem_bytes);

        iterate_range_omp_for(num_groups, [&, f](sycl::id<Dim> group_id) {
          sycl::group<Dim> this_group{group_id, local_size, num_groups};

          f(this_group, reducers...);
        });

        sycl::detail::host_local_memory::release();
      },
      reductions...);
}

template <class HierarchicalDecomposition,
          class Function, int dimensions, typename... Reductions>
inline void parallel_region(Function f,
                            const sycl::range<dimensions> num_groups,
                            const sycl::range<dimensions> group_size,
                            std::size_t num_local_mem_bytes,
                            Reductions... reductions)
{
  static_assert(dimensions > 0 && dimensions <= 3,
                "Only dimensions 1,2,3 are supported");

  reducible_parallel_invocation(
      [&, f, num_groups, group_size](auto &... reducers) {

        sycl::detail::host_local_memory::request_from_threadprivate_pool(
            num_local_mem_bytes);

        iterate_range_omp_for(num_groups, [&](sycl::id<dimensions> group_id) {

          using group_properties =
              sycl::detail::sp_property_descriptor<dimensions, 0,
                                                   HierarchicalDecomposition>;

          sycl::detail::sp_group<
              sycl::detail::host_sp_property_descriptor<group_properties>>
              this_group{
                  sycl::group<dimensions>{group_id, group_size, num_groups}};

          f(this_group, reducers...);
        });

        sycl::detail::host_local_memory::release();

      },
      reductions...);
}

template<int Dim, int MaxGuaranteedWorkgroupSize>
constexpr auto determine_hierarchical_decomposition() {
  using namespace sycl::detail;
  using fallback_decomposition =
      nested_range<unknown_static_range, nested_range<static_range<1>>>;

  if constexpr(Dim == 1) {
    if constexpr(MaxGuaranteedWorkgroupSize % 16 == 0) {
      return nested_range<
        unknown_static_range,
        nested_range<
          static_range<16>
        >
      >{};
    } else {
      return fallback_decomposition{};
    }
  } else if constexpr(Dim == 2){
    if constexpr(MaxGuaranteedWorkgroupSize % 4 == 0) {
      return nested_range<
          unknown_static_range,
          nested_range<
            static_range<4,4>
          >
        >{};
    } else {
      return fallback_decomposition{};
    }
  } else {
    if constexpr(MaxGuaranteedWorkgroupSize % 2 == 0) {
      return nested_range<
          unknown_static_range,
          nested_range<
            static_range<2,2,2>
          >
        >{};
    } else {
      return fallback_decomposition{};
    }
  }
}

}

class omp_kernel_launcher : public rt::backend_kernel_launcher
{
public:

  omp_kernel_launcher() {}
  virtual ~omp_kernel_launcher(){}

  virtual void set_params(void*) override {}

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {

    this->_type = type;
#if !defined(HIPSYCL_HAS_FIBERS) && !defined(__HIPSYCL_USE_ACCELERATED_CPU__)
    if (type == rt::kernel_type::ndrange_parallel_for) {
      this->_invoker = [](rt::dag_node* node) {};

      throw sycl::feature_not_supported{
        "nd_range kernels on CPU are only supported if either compiler support (requires using Clang)\n"
        "or fibers are enabled, as otherwise they cannot be efficiently implemented. It is recommended:\n"
        " * to verify that you really need the features of nd_range parallel for.\n"
        "   If you do not need local memory, use basic parallel for instead.\n"
        " * users targeting SYCL 1.2.1 may use hierarchical parallel for, which\n"
        "   can express the same algorithms, but may have functionality caveats in hipSYCL\n"
        "   and/or other SYCL implementations.\n"
        " * if you use hipSYCL exclusively, you are encouraged to use scoped parallelism:\n"
        "   https://github.com/illuhad/hipSYCL/blob/develop/doc/scoped-parallelism.md\n"
        " * if you can use Clang, enable the compiler support\n"
        "   CMake: -DHIPSYCL_USE_ACCELERATED_CPU=ON, syclcc: --hipsycl-use-accelerated-cpu\n"
        " * if you absolutely need nd_range parallel for and cannot use Clang, enable fiber support in hipSYCL."
      };
    }
#endif

    this->_invoker = [=] (rt::dag_node* node) mutable {

      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k, reductions...);

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      auto get_grid_range = [&]() {
        for (int i = 0; i < Dim; ++i){
          if (global_range[i] % local_range[i] != 0) {
            rt::register_error(__hipsycl_here(),
                               rt::error_info{"omp_dispatch: global range is "
                                              "not divisible by local range"});
          }
        }

        return global_range / local_range;
      };

      if constexpr(type == rt::kernel_type::single_task){

        omp_dispatch::single_task_kernel(k);

      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        if(!is_with_offset) {
          omp_dispatch::parallel_for_kernel(k, global_range, reductions...);
        } else {
          omp_dispatch::parallel_for_kernel_offset(k, global_range, offset, reductions...);
        }

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

        omp_dispatch::parallel_for_ndrange_kernel(
            k, get_grid_range(), local_range, offset, dynamic_local_memory,
            reductions...);

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

        omp_dispatch::parallel_for_workgroup(k, get_grid_range(), local_range,
                                             dynamic_local_memory, reductions...);
      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        auto local_range_is_divisible_by = [&](int x) -> bool {
          for(int i = 0; i < Dim; ++i) {
            if(local_range[i] % x != 0)
              return false;
          }
          return true;
        };

        if(local_range_is_divisible_by(64)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 64>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory,
              reductions...);
        } else if(local_range_is_divisible_by(32)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 32>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory,
              reductions...);
        } else if(local_range_is_divisible_by(16)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 16>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory,
              reductions...);
        } else if(local_range_is_divisible_by(8)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<Dim,
                                                                          8>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory,
              reductions...);
        } else {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<Dim,
                                                                          1>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory,
              reductions...);
        }
      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{
            rt::device_id{rt::backend_descriptor{rt::hardware_platform::cpu,
                                                 rt::api_platform::omp},
                          0},
            static_cast<void*>(nullptr)};

        k(handle);
      }
      else {
        assert(false && "Unsupported kernel type");
      }

    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::omp;
  }

  virtual void invoke(rt::dag_node* node) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:

  std::function<void (rt::dag_node*)> _invoker;
  rt::kernel_type _type;
};

}
}

#endif
