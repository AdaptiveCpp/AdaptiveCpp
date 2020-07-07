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

#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/range.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/nd_item.hpp"
#include "hipSYCL/sycl/group.hpp"
#include "hipSYCL/sycl/detail/local_memory_allocator.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

namespace hipsycl {
namespace glue {
namespace openmp_dispatch {

template<class Function>
inline 
void single_task_kernel(Function f) noexcept
{
  f();
}

template<class Function, class Offset_mode> 
inline
void parallel_for_kernel(Function f,
                        const sycl::range<1> execution_range,
                        Offset_mode,
                        const sycl::id<1> offset = sycl::id<1>{}) noexcept
{
  const size_t max_id = offset.get(0) + execution_range.get(0);

#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel for
#endif
  for(size_t i = offset.get(0); i < max_id; ++i)
  {
    auto this_item = 
      make_item_maybe_with_offset(sycl::id<1>{i}, 
                                  execution_range,
                                  offset, Offset_mode{});

    f(this_item);
  }
}

template<class Function, class Offset_mode> 
inline 
void parallel_for_kernel(Function f,
                        const sycl::range<2> execution_range,
                        Offset_mode,
                        const sycl::id<2> offset = sycl::id<2>{}) noexcept
{
  const sycl::id<2> max_id = offset + execution_range;

#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel for collapse(2)
#endif
  for(size_t i = offset.get(0); i < max_id.get(0); ++i)
    for(size_t j = offset.get(1); j < max_id.get(1); ++j)
    {
      auto this_item = 
        make_item_maybe_with_offset(sycl::id<2>{i,j}, 
                                    execution_range, 
                                    offset, Offset_mode{});

      f(this_item);
    }
}

template<class Function, class Offset_mode> 
inline 
void parallel_for_kernel(Function f,
                        const sycl::range<3> execution_range,
                        Offset_mode,
                        const sycl::id<3> offset = sycl::id<3>{}) noexcept
{
  const sycl::id<3> max_id = offset + execution_range;

#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel for collapse(3)
#endif
  for(size_t i = offset.get(0); i < max_id.get(0); ++i)
    for(size_t j = offset.get(1); j < max_id.get(1); ++j)
      for(size_t k = offset.get(2); k < max_id.get(2); ++k)
    {
      auto this_item = 
        make_item_maybe_with_offset(sycl::id<3>{i,j,k}, 
                                    execution_range, 
                                    offset, Offset_mode{});
        
      f(this_item);
    }
}

// This must still be executed by hipCPU's hipLaunchKernel for correct
// synchronization semantics
template<int dimensions, class Function>
inline
void parallel_for_ndrange_kernel(Function f, sycl::id<dimensions> offset) noexcept
{
  sycl::nd_item<dimensions> this_item{
    &offset,
    sycl::detail::get_group_id<dimensions>(),
    sycl::detail::get_local_id<dimensions>(),
    sycl::detail::get_local_size<dimensions>(),
    sycl::detail::get_grid_size<dimensions>()
  };

  sycl::detail::host_local_memory::request_from_hipcpu_pool();
  f(this_item);
  sycl::detail::host_local_memory::release();
}

template<class Function>
inline 
void parallel_for_workgroup(Function f,
                            const sycl::range<1> num_groups,
                            const sycl::range<1> local_size,
                            size_t num_local_mem_bytes) noexcept
{
#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel
#endif
  {
    sycl::detail::host_local_memory::request_from_threadprivate_pool(num_local_mem_bytes);

#ifndef HIPCPU_NO_OPENMP
    #pragma omp for
#endif
    for(size_t i = 0; i < num_groups.get(0); ++i)
    {
      sycl::group<1> this_group{sycl::id<1>{i}, local_size, num_groups};
      f(this_group);
    }

    sycl::detail::host_local_memory::release();
  }
}

template<class Function>
inline 
void parallel_for_workgroup(Function f,
                            const sycl::range<2> num_groups,
                            const sycl::range<2> local_size,
                            size_t num_local_mem_bytes) noexcept
{
#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel
#endif
  {
    host_local_memory::request_from_threadprivate_pool(num_local_mem_bytes);

#ifndef HIPCPU_NO_OPENMP
    #pragma omp for collapse(2)
#endif
    for(size_t i = 0; i < num_groups.get(0); ++i)
      for(size_t j = 0; j < num_groups.get(1); ++j)
      {
        group<2> this_group{sycl::id<2>{i,j}, local_size, num_groups};
        f(this_group);
      }

    host_local_memory::release();
  }
}

template<class Function>
inline 
void parallel_for_workgroup(Function f,
                            const sycl::range<3> num_groups,
                            const sycl::range<3> local_size,
                            size_t num_local_mem_bytes) noexcept
{
#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel
#endif
  {
    host_local_memory::request_from_threadprivate_pool(num_local_mem_bytes);

#ifndef HIPCPU_NO_OPENMP
    #pragma omp for collapse(3)
#endif
    for(size_t i = 0; i < num_groups.get(0); ++i)
      for(size_t j = 0; j < num_groups.get(1); ++j)
        for(size_t k = 0; k < num_groups.get(2); ++k)
        {
          group<3> this_group{sycl::id<3>{i,j,k}, local_size, num_groups};
          f(this_group);
        }
    
    host_local_memory::release();
  }
}

template<class Function, int dimensions>
inline 
void parallel_region(Function f,
                    sycl::range<dimensions> num_groups,
                    sycl::range<dimensions> group_size,
                    std::size_t num_local_mem_bytes)
{
#ifndef HIPCPU_NO_OPENMP
  #pragma omp parallel
#endif
  {
    host_local_memory::request_from_threadprivate_pool(num_local_mem_bytes);

    auto make_physical_item = [&](sycl::id<dimensions> group_id){
      return detail::make_sp_item(sycl::id<dimensions>{},group_id,group_size, num_groups);
    };

    if constexpr(dimensions == 1){
#ifndef HIPCPU_NO_OPENMP
    #pragma omp for
#endif
      for(size_t i = 0; i < num_groups.get(0); ++i){
        auto group_id = sycl::id<1>{i};
        group<1> this_group{group_id, group_size, num_groups};
        f(this_group, make_physical_item(group_id));
      }

    } else if constexpr(dimensions == 2){
#ifndef HIPCPU_NO_OPENMP
    #pragma omp for collapse(2)
#endif
      for(size_t i = 0; i < num_groups.get(0); ++i)
        for(size_t j = 0; j < num_groups.get(1); ++j)
        {
          auto group_id = sycl::id<2>{i,j};
          group<2> this_group{group_id, group_size, num_groups};
          f(this_group, make_physical_item(group_id));
        } 
    } else {
#ifndef HIPCPU_NO_OPENMP
    #pragma omp for collapse(3)
#endif
      for(size_t i = 0; i < num_groups.get(0); ++i)
        for(size_t j = 0; j < num_groups.get(1); ++j)
          for(size_t k = 0; k < num_groups.get(2); ++k)
          {
            auto group_id = sycl::id<3>{i,j,k};
            group<3> this_group{group_id, group_size, num_groups};
            f(this_group, make_physical_item(group_id));
          }

    }    
    host_local_memory::release();
  }
}

}

class openmp_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  openmp_kernel_launcher()
      : _queue{nullptr}{}


  void set_params() {}

  template <class KernelName, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k) {

    _invoker = [=]() {
      assert(_queue != nullptr);

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      if constexpr(type == rt::kernel_type::single_task){
        
      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        if(!is_with_offset) {
        } else {
        }
        
      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {
        
        for (int i = 0; i < Dim; ++i)
          assert(global_range[i] % local_range[i] == 0);

        sycl::range<Dim> grid_range = global_range / local_range;
        
        
      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        for (int i = 0; i < Dim; ++i)
          assert(global_range[i] % local_range[i] == 0);

        sycl::range<Dim> grid_range = global_range / local_range;

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

  virtual ~openmp_kernel_launcher(){}
private:
  std::function<void ()> _invoker;
};

}
}

#endif