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

#ifndef HIPSYCL_HIPLIKE_KERNEL_LAUNCHER_HPP
#define HIPSYCL_HIPLIKE_KERNEL_LAUNCHER_HPP

#include <cassert>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

#include "clang.hpp"


namespace hipsycl {
namespace glue {

namespace hiplike_dispatch {

template<int dimensions>
__host__ __device__
bool item_is_in_range(const sycl::item<dimensions, true>& item,
                      const sycl::range<dimensions>& execution_range,
                      const sycl::id<dimensions>& offset)
{
  for(int i = 0; i < dimensions; ++i)
  {
    if(item.get_id(i) >= offset.get(i) + execution_range.get(i))
    {
      return false;
    }
  }
  return true;
}

template<int dimensions>
__host__ __device__
bool item_is_in_range(const sycl::item<dimensions, false>& item,
                      const sycl::range<dimensions>& execution_range)
{
  for(int i = 0; i < dimensions; ++i)
  {
    if(item.get_id(i) >= execution_range.get(i))
    {
      return false;
    }
  }
  return true;
}

template<class F>
__host__ __device__
void device_invocation(F f)
{
#ifdef SYCL_DEVICE_ONLY
  f();
#else
  assert(false && "Attempted to execute device code path on host!");
#endif
}

template<typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f)
{
  device_invocation(f);
}

template <typename KernelName, class Function, int dimensions>
__sycl_kernel void
parallel_for_kernel(Function f, sycl::range<dimensions> execution_range,
                    sycl::id<dimensions> offset, bool with_offset) {
  // Note: We currently cannot have with_offset as template parameter
  // because this might cause clang to emit two kernels with the same
  // mangled name (variants with and without offset) if an explicit kernel
  // name is provided.
  if(with_offset) {
    device_invocation([&] __host__ __device__() {
      auto this_item = sycl::detail::make_item<dimensions>(
          sycl::detail::get_global_id<dimensions>() + offset, execution_range,
          offset);
      if (item_is_in_range(this_item, execution_range, offset))
        f(this_item);
    });
  } else {
    device_invocation([&] __host__ __device__() {
      auto this_item = sycl::detail::make_item<dimensions>(
          sycl::detail::get_global_id<dimensions>(), execution_range);
      if (item_is_in_range(this_item, execution_range))
        f(this_item);
    });
  }
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel
void parallel_for_ndrange_kernel(Function f, sycl::id<dimensions> offset)
{
  device_invocation([&] __host__ __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    sycl::nd_item<dimensions> this_item{&offset};
#else
    sycl::nd_item<dimensions> this_item{
        &offset,
        sycl::detail::get_group_id<dimensions>(),
        sycl::detail::get_local_id<dimensions>(),
        sycl::detail::get_local_size<dimensions>(),
        sycl::detail::get_grid_size<dimensions>()};
#endif
    f(this_item);
  });
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_workgroup(Function f,
                            // The logical group size is not yet used,
                            // but it's still useful to already have it here
                            // since it allows the compiler to infer 'dimensions'
                            sycl::range<dimensions> logical_group_size)
{
  device_invocation([&] __host__ __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    sycl::group<dimensions> this_group;
#else
    sycl::group<dimensions> this_group{
        sycl::detail::get_group_id<dimensions>(),
        sycl::detail::get_local_size<dimensions>(),
        sycl::detail::get_grid_size<dimensions>()};
#endif
    f(this_group);
  });
}


template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_region(Function f,
                    sycl::range<dimensions> num_groups,
                    sycl::range<dimensions> group_size)
{
  device_invocation([&] __host__ __device__ () {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    sycl::group<dimensions> this_group;
#else
    sycl::group<dimensions> this_group{
      sycl::detail::get_group_id<dimensions>(), 
      sycl::detail::get_local_size<dimensions>(),
      sycl::detail::get_grid_size<dimensions>()
    };
#endif
    sycl::physical_item<dimensions> phys_idx = sycl::detail::make_sp_item(
      sycl::detail::get_local_id<dimensions>(),
      sycl::detail::get_group_id<dimensions>(),
      sycl::detail::get_local_size<dimensions>(),
      sycl::detail::get_grid_size<dimensions>()
    );
    
    f(this_group, phys_idx);
  });
}


/// Flips dimensions such that the range is consistent with the mapping
/// of SYCL index dimensions to backend dimensions.
/// When launching a SYCL kernel, grid and blocksize should be transformed
/// using this function.
template<int dimensions>
inline dim3 make_kernel_launch_range(dim3 range);

template<>
inline dim3 make_kernel_launch_range<1>(dim3 range)
{
  return dim3(range.x, 1, 1);
}

template<>
inline dim3 make_kernel_launch_range<2>(dim3 range)
{
  return dim3(range.y, range.x, 1);
}

template<>
inline dim3 make_kernel_launch_range<3>(dim3 range)
{
  return dim3(range.z, range.y, range.x);
}

template <int dimensions>
inline dim3 make_kernel_launch_range(sycl::range<dimensions> r) {
  if(dimensions == 1)
    return make_kernel_launch_range<dimensions>(dim3(r[0], 1, 1));
  else if (dimensions == 2)
    return make_kernel_launch_range<dimensions>(dim3(r[0], r[1], 1));
  else if (dimensions == 3)
    return make_kernel_launch_range<dimensions>(dim3(r[0], r[1], r[2]));
  return dim3(1,1,1);
}

inline std::size_t ceil_division(std::size_t n, std::size_t divisor) {
  return (n + divisor - 1) / divisor;
}

template <int dimensions>
inline sycl::range<dimensions>
determine_grid_configuration(const sycl::range<dimensions> &num_work_items,
                             const sycl::range<dimensions> &local_range) {
  sycl::range<dimensions> res;
  for (int i = 0; i < dimensions; ++i)
    res[i] = ceil_division(num_work_items[i], local_range[i]);
  return res;
}


} // hiplike_dispatch

template<rt::backend_id Backend_id, class Queue_type>
class hiplike_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  hiplike_kernel_launcher()
      : _queue{nullptr}{}


  virtual void set_params(void *q) override {
    _queue = reinterpret_cast<Queue_type*>(q);
  }

  Queue_type *get_queue() const {
    return _queue;
  }

  template <class KernelName, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {
    
    this->_type = type;

    _invoker = [=]() {
      assert(_queue != nullptr);

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      if constexpr (type == rt::kernel_type::single_task) {
        __hipsycl_launch_kernel(
            hiplike_dispatch::single_task_kernel<KernelName>, 1, 1,
            dynamic_local_memory, _queue->get_stream(), k);
      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        sycl::range<Dim> local_range;
        if constexpr (Dim == 1)
          local_range = sycl::range<1>{128};
        else if constexpr (Dim == 2)
          local_range = sycl::range<2>{16, 16};
        else if constexpr (Dim == 3)
          local_range = sycl::range<3>{4, 8, 8};

        sycl::range<Dim> grid_range =
            hiplike_dispatch::determine_grid_configuration(global_range,
                                                            local_range);

        __hipsycl_launch_kernel(
            hiplike_dispatch::parallel_for_kernel<KernelName>,
            hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
            hiplike_dispatch::make_kernel_launch_range<Dim>(local_range),
            dynamic_local_memory, _queue->get_stream(), k, global_range,
            offset, is_with_offset);

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

        for (int i = 0; i < Dim; ++i)
          assert(global_range[i] % local_range[i] == 0);

        sycl::range<Dim> grid_range = global_range / local_range;

        __hipsycl_launch_kernel(
            hiplike_dispatch::parallel_for_ndrange_kernel<KernelName>,
            hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
            hiplike_dispatch::make_kernel_launch_range<Dim>(local_range),
            dynamic_local_memory, _queue->get_stream(), k, offset);

      } else if constexpr (type ==
                            rt::kernel_type::hierarchical_parallel_for) {

        for (int i = 0; i < Dim; ++i)
          assert(global_range[i] % local_range[i] == 0);

        sycl::range<Dim> grid_range = global_range / local_range;

        __hipsycl_launch_kernel(
            hiplike_dispatch::parallel_for_workgroup<KernelName>,
            hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
            hiplike_dispatch::make_kernel_launch_range<Dim>(local_range),
            dynamic_local_memory, _queue->get_stream(), k, local_range);

      } else if constexpr (type == rt::kernel_type::scoped_parallel_for) {

        for (int i = 0; i < Dim; ++i)
          assert(global_range[i] % local_range[i] == 0);

        sycl::range<Dim> grid_range = global_range / local_range;

        __hipsycl_launch_kernel(
            hiplike_dispatch::parallel_region<KernelName>,
            hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
            hiplike_dispatch::make_kernel_launch_range<Dim>(local_range),
            dynamic_local_memory, _queue->get_stream(), k, grid_range,
            local_range);
      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{_queue->get_device(),
                                    static_cast<void *>(_queue)};

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
    return Backend_id;
  }

  virtual void invoke() final override {
    _invoker();
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

  virtual ~hiplike_kernel_launcher() {}
  
private:
  Queue_type *_queue;
  rt::kernel_type _type;
  std::function<void ()> _invoker;
};

}
}

#endif
