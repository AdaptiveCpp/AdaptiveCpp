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
#include <omp.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

#include "../generic/host/collective_execution_engine.hpp"
#include "../generic/host/iterate_range.hpp"

namespace hipsycl {
namespace glue {
namespace omp_dispatch {


template <int Dim, class Function>
void iterate_range_omp_for(sycl::range<Dim> r, Function f) {

  if constexpr (Dim == 1) {
    #pragma omp for
    for (std::size_t i = 0; i < r.get(0); ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    #pragma omp for collapse(2)
    for (std::size_t i = 0; i < r.get(0); ++i) {
      for (std::size_t j = 0; j < r.get(1); ++j) {
        f(sycl::id<Dim>{i, j});
      }
    }
  } else if constexpr (Dim == 3) {
    #pragma omp for collapse(3)
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
                           Function f) {

  const std::size_t min_i = offset.get(0);
  const std::size_t max_i = offset.get(0) + r.get(0);

  if constexpr (Dim == 1) {
  #pragma omp for
    for (std::size_t i = min_i; i < max_i; ++i) {
      f(sycl::id<Dim>{i});
    }
  } else if constexpr (Dim == 2) {
    const std::size_t min_j = offset.get(1);
    const std::size_t max_j = offset.get(1) + r.get(1);

  #pragma omp for collapse(2)
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

  #pragma omp for collapse(3)
    for (std::size_t i = min_i; i < max_i; ++i) {
      for (std::size_t j = min_j; j < max_j; ++j) {
        for (std::size_t k = min_k; k < max_k; ++k) {
          f(sycl::id<Dim>{i, j, k});
        }
      }
    }
  }
}

template<class Function>
inline 
void single_task_kernel(Function f) noexcept
{
  f();
}

template<int Dim, class Function> 
inline 
void parallel_for_kernel(Function f,
                        const sycl::range<Dim> execution_range) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

#pragma omp parallel
  {
    iterate_range_omp_for(execution_range, [&](sycl::id<Dim> idx) {
      auto this_item = 
        sycl::detail::make_item<Dim>(idx, execution_range);

      f(this_item);
    });
  }
}

template<int Dim, class Function> 
inline 
void parallel_for_kernel(Function f,
                        const sycl::range<Dim> execution_range,
                        const sycl::id<Dim> offset) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");


#pragma omp parallel
  {
    iterate_range_omp_for(offset, execution_range, [&](sycl::id<Dim> idx) {
      auto this_item = 
        sycl::detail::make_item<Dim>(idx, execution_range, offset);

      f(this_item);
    });
  }
}


template <int Dim, class Function>
inline void parallel_for_ndrange_kernel(
    Function f, const sycl::range<Dim> num_groups,
    const sycl::range<Dim> local_size, sycl::id<Dim> offset,
    size_t num_local_mem_bytes) noexcept
{

#ifndef HIPSYCL_HAS_FIBERS
  rt::register_error(__hipsycl_here(),
                     rt::error_info{
                         "nd_range parallel for on CPU requires fibers, but "
                         "fiber support is disabled",
                     rt::error_type::feature_not_supported});
#else
  static_assert(Dim > 0 && Dim <= 3,
                "Only dimensions 1 - 3 are supported.");

#pragma omp parallel
  {
    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);


    host::static_range_decomposition<Dim> group_decomposition{
          num_groups, omp_get_num_threads()};

    host::collective_execution_engine<Dim> engine{num_groups, local_size,
                                                  offset, group_decomposition,
                                                  omp_get_thread_num()};

    std::function<void()> barrier_impl = [&]() { engine.barrier(); };

    engine.run_kernel([&](sycl::id<Dim> local_id, sycl::id<Dim> group_id) {
      sycl::nd_item<Dim> this_item{&offset,    group_id,   local_id,
                                   local_size, num_groups, &barrier_impl};

      f(this_item);
    });

    sycl::detail::host_local_memory::release();
  }
#endif
}

template<int Dim, class Function>
inline 
void parallel_for_workgroup(Function f,
                            const sycl::range<Dim> num_groups,
                            const sycl::range<Dim> local_size,
                            size_t num_local_mem_bytes) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

#pragma omp parallel
  {
    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    iterate_range_omp_for(num_groups, [&](sycl::id<Dim> group_id) {
      sycl::group<Dim> this_group{group_id, local_size, num_groups};
      f(this_group);
    });
 
    sycl::detail::host_local_memory::release();
  }
}

template<class Function, int dimensions>
inline 
void parallel_region(Function f,
                    sycl::range<dimensions> num_groups,
                    sycl::range<dimensions> group_size,
                    std::size_t num_local_mem_bytes)
{
  static_assert(dimensions > 0 && dimensions <= 3,
                "Only dimensions 1,2,3 are supported");

#pragma omp parallel
  {
    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    iterate_range_omp_for(num_groups, [&](sycl::id<dimensions> group_id) {
      sycl::group<dimensions> this_group{group_id, group_size, num_groups};

      auto phys_item = sycl::detail::make_sp_item(
          sycl::id<dimensions>{}, group_id, group_size, num_groups);

      f(this_group, phys_item);
    });
 
    sycl::detail::host_local_memory::release();
  }
}

}

class omp_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  omp_kernel_launcher() {}
  virtual ~omp_kernel_launcher(){}

  virtual void set_params(void*) override {}

  template <class KernelName, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {

    this->_type = type;

#ifndef HIPSYCL_HAS_FIBERS
    if (type == rt::kernel_type::ndrange_parallel_for) {
      this->_invoker = []() {};
      
      throw sycl::feature_not_supported{
        "Support for nd_range kernels on CPU is disabled without fibers because they cannot be\n"
        "implemented efficiently in pure-library SYCL implementations such as hipSYCL on CPU. We recommend:\n"
        " * to verify that you really need the features of nd_range parallel for.\n"
        "   If you do not need local memory, use basic parallel for instead.\n"
        " * users targeting SYCL 1.2.1 may use hierarchical parallel for, which\n"
        "   can express the same algorithms, but may have functionality caveats in hipSYCL\n"
        "   and/or other SYCL implementations.\n"
        " * if you use hipSYCL exclusively, you are encouraged to use scoped parallelism:\n"
        "   https://github.com/illuhad/hipSYCL/blob/develop/doc/scoped-parallelism.md\n"
        " * if you absolutely need nd_range parallel for, enable fiber support in hipSYCL."
      };
    }
#endif
    
    this->_invoker = [=]() {

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
          omp_dispatch::parallel_for_kernel(k, global_range);
        } else {
          omp_dispatch::parallel_for_kernel(k, global_range, offset);
        }

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

        omp_dispatch::parallel_for_ndrange_kernel(k, get_grid_range(), local_range,
                                                  offset, dynamic_local_memory);

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

        omp_dispatch::parallel_for_workgroup(k, get_grid_range(), local_range,
                                             dynamic_local_memory);
      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        omp_dispatch::parallel_region(k, get_grid_range(), local_range,
                                      dynamic_local_memory);
      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{
            rt::device_id{rt::backend_descriptor{rt::hardware_platform::cpu,
                                                 rt::api_platform::omp},
                          0},
            static_cast<void*>(nullptr)};

        // Need to perform additional copy to guarantee deferred_pointers/
        // accessors are initialized
        auto initialized_kernel_invoker = k;
        initialized_kernel_invoker(handle);
      }
      else {
        assert(false && "Unsupported kernel type");
      }
      
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::omp;
  }

  virtual void invoke() final override {
    _invoker();
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:

  std::function<void ()> _invoker;
  rt::kernel_type _type;
};

}
}

#endif