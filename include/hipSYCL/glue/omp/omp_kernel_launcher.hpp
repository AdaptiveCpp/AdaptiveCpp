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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/range.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/nd_item.hpp"
#include "hipSYCL/sycl/sp_item.hpp"
#include "hipSYCL/sycl/group.hpp"
#include "hipSYCL/sycl/detail/local_memory_allocator.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

namespace hipsycl {
namespace glue {
namespace omp_dispatch {

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

  const auto max_i = execution_range.get(0);

  if constexpr(Dim == 1){
#pragma omp parallel for
    for(size_t i = 0; i < max_i; ++i){
      auto this_item = 
        sycl::detail::make_item<Dim>(sycl::id<1>{i}, 
                                    execution_range);

      f(this_item);
    }
  }
  else if constexpr(Dim == 2){
    const auto max_j = execution_range.get(1);
#pragma omp parallel for collapse(2)
    for(size_t i = 0; i < max_i; ++i)
      for(size_t j = 0; j < max_j; ++j)
      {
        auto this_item = 
          sycl::detail::make_item<Dim>(sycl::id<2>{i,j}, 
                                      execution_range);

        f(this_item);
      }
  }
  else if constexpr(Dim == 3){
    const auto max_j = execution_range.get(1);
    const auto max_k = execution_range.get(2);
#pragma omp parallel for collapse(3)
    for(size_t i = 0; i < max_i; ++i)
      for(size_t j = 0; j < max_j; ++j)
        for(size_t k = 0; k < max_k; ++k)
        {
          auto this_item = 
            sycl::detail::make_item<Dim>(sycl::id<3>{i,j,k}, 
                                        execution_range);

          f(this_item);
        }
  }
}

template<int Dim, class Function> 
inline 
void parallel_for_kernel(Function f,
                        const sycl::range<Dim> execution_range,
                        const sycl::id<Dim> offset) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

  const auto max_i = execution_range.get(0);
  const auto min_i = offset.get(0);

  if constexpr(Dim == 1){
#pragma omp parallel for
    for(size_t i = min_i; i < max_i; ++i){
      auto this_item = 
        sycl::detail::make_item<Dim>(sycl::id<1>{i}, 
                                    execution_range, offset);

      f(this_item);
    }
  }
  else if constexpr(Dim == 2){
    const auto max_j = execution_range.get(1);
    const auto min_j = offset.get(1);

#pragma omp parallel for collapse(2)
    for(size_t i = min_i; i < max_i; ++i)
      for(size_t j = min_j; j < max_j; ++j)
      {
        auto this_item = 
          sycl::detail::make_item<Dim>(sycl::id<2>{i,j}, 
                                      execution_range, offset);

        f(this_item);
      }
  }
  else if constexpr(Dim == 3){
    const auto max_j = execution_range.get(1);
    const auto max_k = execution_range.get(2);
    const auto min_j = offset.get(1);
    const auto min_k = offset.get(2);
#pragma omp parallel for collapse(3)
    for(size_t i = min_i; i < max_i; ++i)
      for(size_t j = min_j; j < max_j; ++j)
        for(size_t k = min_k; k < max_k; ++k)
        {
          auto this_item = 
            sycl::detail::make_item<Dim>(sycl::id<3>{i,j,k}, 
                                        execution_range, offset);

          f(this_item);
        }
  }
}

template<int Dim, class Function>
inline
void parallel_for_ndrange_kernel(Function f, 
                            const sycl::range<Dim> num_groups,
                            const sycl::range<Dim> local_size,
                            sycl::id<Dim> offset,
                            size_t num_local_mem_bytes) noexcept
{
  rt::register_error(
      __hipsycl_here(),
      rt::error_info{"nd_range parallel for is unsupported on CPU because it "
                     "cannot be implemented efficiently.",
                     rt::error_type::feature_not_supported});
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

    if constexpr(Dim == 1){
#pragma omp for
      for(size_t i = 0; i < num_groups.get(0); ++i){
        sycl::group<1> this_group{sycl::id<1>{i}, local_size, num_groups};
        f(this_group);
      }
    }
    else if constexpr(Dim == 2){
#pragma omp for collapse(2)
      for(size_t i = 0; i < num_groups.get(0); ++i){
        for(size_t j = 0; j < num_groups.get(1); ++j){
          sycl::group<2> this_group{sycl::id<2>{i,j}, local_size, num_groups};
          f(this_group);
        }
      }
    }
    else if constexpr(Dim == 3){
#pragma omp for collapse(3)
      for(size_t i = 0; i < num_groups.get(0); ++i){
        for(size_t j = 0; j < num_groups.get(1); ++j){
          for(size_t k = 0; k < num_groups.get(2); ++k){
            sycl::group<3> this_group{sycl::id<3>{i,j,k}, local_size, num_groups};
            f(this_group);
          }
        }
      }
    }

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

    auto make_physical_item = [&](sycl::id<dimensions> group_id) {
      return sycl::detail::make_sp_item(sycl::id<dimensions>{}, group_id,
                                        group_size, num_groups);
    };

    if constexpr(dimensions == 1){
    #pragma omp for
      for(size_t i = 0; i < num_groups.get(0); ++i){
        auto group_id = sycl::id<1>{i};
        sycl::group<1> this_group{group_id, group_size, num_groups};
        f(this_group, make_physical_item(group_id));
      }

    } else if constexpr(dimensions == 2){
    #pragma omp for collapse(2)
      for(size_t i = 0; i < num_groups.get(0); ++i)
        for(size_t j = 0; j < num_groups.get(1); ++j)
        {
          auto group_id = sycl::id<2>{i,j};
          sycl::group<2> this_group{group_id, group_size, num_groups};
          f(this_group, make_physical_item(group_id));
        } 
    } else {
    #pragma omp for collapse(3)
      for(size_t i = 0; i < num_groups.get(0); ++i)
        for(size_t j = 0; j < num_groups.get(1); ++j)
          for(size_t k = 0; k < num_groups.get(2); ++k)
          {
            auto group_id = sycl::id<3>{i,j,k};
            sycl::group<3> this_group{group_id, group_size, num_groups};
            f(this_group, make_physical_item(group_id));
          }

    }    
    sycl::detail::host_local_memory::release();
  }
}

}

class omp_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  omp_kernel_launcher() {}
  virtual ~omp_kernel_launcher(){}

  void set_params() {}

  template <class KernelName, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k) {
    
    this->_type = type;
    
    if(type == rt::kernel_type::ndrange_parallel_for) {
      throw sycl::feature_not_supported{
        "The CPU backend does not support nd_range kernels because they cannot be implemented\n"
        "efficiently in pure-library SYCL implementations such as hipSYCL on CPU. We recommend:\n"
        " * to verify that you really need the features of nd_range parallel for.\n"
        "   If you do not need local memory, use basic parallel for instead.\n"
        " * users targeting SYCL 1.2.1 may use hierarchical parallel for, which\n"
        "   can express the same algorithms, but may have functionality caveats in hipSYCL\n"
        "   and/or other SYCL implementations.\n"
        " * if you use hipSYCL exclusively, you are encouraged to use scoped parallelism:\n"
        "   https://github.com/illuhad/hipSYCL/blob/develop/doc/scoped-parallelism.md"
      };
    }

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

      }
      else {
        assert(false && "Unsupported kernel type");
      }
      
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::openmp_cpu;
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