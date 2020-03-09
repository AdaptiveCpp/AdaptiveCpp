/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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


#ifndef HIPSYCL_HANDLER_HPP
#define HIPSYCL_HANDLER_HPP

#include <type_traits>
#include <unordered_map>

#include "exception.hpp"
#include "access.hpp"
#include "accessor.hpp"
#include "backend/backend.hpp"
#include "types.hpp"
#include "id.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "item.hpp"
#include "nd_item.hpp"
#include "group.hpp"
#include "detail/local_memory_allocator.hpp"
#include "detail/buffer.hpp"
#include "detail/task_graph.hpp"
#include "detail/application.hpp"
#include "detail/stream.hpp"
#include "detail/debug.hpp"
#include "detail/util.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {


namespace dispatch {


template<int dimensions, bool with_offset>
HIPSYCL_KERNEL_TARGET
bool item_is_in_range(const sycl::item<dimensions, with_offset>& item,
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


namespace device {

// These helper functions are necessary because we cannot
// call device functions directly from __sycl_kernel functions
// with the clang plugin, since they are initially treated
// as host functions.
// TODO: Find a nicer solution for that.
template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_global_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_global_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_local_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_local_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_group_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_group_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::range<dimensions> get_local_size_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_local_size<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::range<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::range<dimensions> get_grid_size_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_grid_size<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::range<dimensions>>();
#endif
}

template<typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f)
{
  f();
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel(Function f,
                        sycl::range<dimensions> execution_range)
{
  auto this_item = detail::make_item<dimensions>(
    get_global_id_helper<dimensions>(), execution_range);
  if(item_is_in_range(this_item, execution_range, sycl::id<dimensions>{}))
    f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel_with_offset(Function f,
                                    sycl::range<dimensions> execution_range,
                                    sycl::id<dimensions> offset)
{
  auto this_item = detail::make_item<dimensions>(
    get_global_id_helper<dimensions>() + offset, execution_range, offset);
  if(item_is_in_range(this_item, execution_range, offset))
    f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel
void parallel_for_ndrange_kernel(Function f, sycl::id<dimensions> offset)
{
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
  nd_item<dimensions> this_item{&offset};
#else
  nd_item<dimensions> this_item{
    &offset, 
    get_group_id_helper<dimensions>(), 
    get_local_id_helper<dimensions>(),
    get_local_size_helper<dimensions>(),
    get_grid_size_helper<dimensions>()
  };
#endif
  f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_workgroup(Function f,
                            // The logical group size is not yet used,
                            // but it's still useful to already have it here
                            // since it allows the compiler to infer 'dimensions'
                            sycl::range<dimensions> logical_group_size)
{
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
  group<dimensions> this_group;
#else
  group<dimensions> this_group{
    get_group_id_helper<dimensions>(), 
    get_local_size_helper<dimensions>(),
    get_grid_size_helper<dimensions>()
  };
#endif
  f(this_group);
}

} // device

namespace host {

#ifdef HIPSYCL_PLATFORM_CPU

namespace offset{
struct with_offset{};
struct without_offset{};
}

template<int dimensions>
inline item<dimensions, true> make_item_maybe_with_offset(
                    sycl::id<dimensions> effective_id, 
                    sycl::range<dimensions> size,
                    sycl::id<dimensions> offset,
                    offset::with_offset)
{
  return make_item(effective_id, size, offset);
}

template<int dimensions>
inline item<dimensions, true> make_item_maybe_with_offset(
                    sycl::id<dimensions> effective_id, 
                    sycl::range<dimensions> size,
                    sycl::id<dimensions> offset,
                    offset::without_offset)
{
  return make_item(effective_id, size);
}


// On CPU, we can use references for the kernel lambdas. This is because
// they are already copied as a closure to the task graph, which already
// stores all necessary captures etc until the lambda is executed.
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
  nd_item<dimensions> this_item{
    &offset,
    detail::get_group_id<dimensions>(),
    detail::get_local_id<dimensions>(),
    detail::get_local_size<dimensions>(),
    detail::get_grid_size<dimensions>()
  };

  host_local_memory::request_from_hipcpu_pool();
  f(this_item);
  host_local_memory::release();
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
    host_local_memory::request_from_threadprivate_pool(num_local_mem_bytes);

#ifndef HIPCPU_NO_OPENMP
    #pragma omp for
#endif
    for(size_t i = 0; i < num_groups.get(0); ++i)
    {
      group<1> this_group{sycl::id<1>{i}, local_size, num_groups};
      f(this_group);
    }

    host_local_memory::release();
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

#endif // HIPSYCL_PLATFORM_CPU
} // host

} // dispatch

template <access::target srcTgt, access::target dstTgt>
constexpr hipMemcpyKind get_copy_kind()
{
  if(srcTgt == access::target::global_buffer)
  {
    if(dstTgt == access::target::global_buffer) return hipMemcpyDeviceToDevice;
    if(dstTgt == access::target::host_buffer) return hipMemcpyDeviceToHost;
  }
  if(srcTgt == access::target::host_buffer)
  {
    if(dstTgt == access::target::global_buffer) return hipMemcpyHostToDevice;
    if(dstTgt == access::target::host_buffer) return hipMemcpyHostToHost;
  }
  assert(false && "Unsupported / unimplemented access targets");
  return hipMemcpyDefault;
}

/// Used by the command group handler to associate accessors with their buffers.
class accessor_buffer_mapper
{
public:
  accessor_id insert(buffer_ptr buff)
  {
    accessor_id id = static_cast<accessor_id>(_acc_data.size());
    _acc_data[id] = buff;
    return id;
  }

  buffer_ptr find_accessor(accessor_id id) const
  {
    return _acc_data.at(id);
  }

  template<
    class T, 
    int D, 
    access::mode Mode, 
    access::target Target,
    access::placeholder IsPlaceholder,
    HIPSYCL_ENABLE_IF_ACCESSOR_STORES_BUFFER_PTR(Target, IsPlaceholder)>
  buffer_ptr find_accessor(
    const sycl::accessor<T, D, Mode, Target, IsPlaceholder>& acc) const
  {
#ifndef SYCL_DEVICE_ONLY
    // This function is not available when compiling for device,
    // so we need preprocessor guards.
    return acc._buff.get_shared_ptr();
#else
    // Silences warning about missing return value
    return buffer_ptr{};
#endif
  }

  template<
    class T, 
    int D, 
    access::mode Mode, 
    access::target Target,
    access::placeholder IsPlaceholder,
    HIPSYCL_ENABLE_IF_ACCESSOR_STORES_ACCESSOR_ID(Target, IsPlaceholder)>
  buffer_ptr find_accessor(
    const sycl::accessor<T, D, Mode, Target, IsPlaceholder>& acc) const
  {
    return find_accessor(acc._accessor_id);
  }
private:
  std::unordered_map<accessor_id, buffer_ptr> _acc_data;
};

} // detail

class queue;

class handler
{
public:
  ~handler()
  {
  }

  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
  void require(accessor<dataT, dimensions, accessMode, accessTarget,
                        access::placeholder::true_t> acc)
  {
    static_assert(accessTarget == access::target::global_buffer ||
                  accessTarget == access::target::constant_buffer,
                  "Only placeholder accessors for global and constant buffers are "
                  "supported.");

    detail::buffer_ptr buff = _accessor_buffer_map.find_accessor(acc);


    detail::accessor::obtain_device_access(buff,
                                           *this,
                                           accessMode);

  }

  //----- OpenCL interoperability interface is not supported
  /*

template <typename T>
void set_arg(int argIndex, T && arg);

template <typename... Ts>
void set_args(Ts &&... args);
*/
  //------ Kernel dispatch API


  template <typename KernelName = class _unnamed_kernel, typename KernelType>
  void single_task(KernelType kernelFunc)
  {
    // TODO If shared_mem_size != 0, we can raise an error -
    // local memory doesn't make sense for single_task!
    std::size_t shared_mem_size = _local_mem_allocator.get_allocation_size();
    detail::stream_ptr stream = this->get_stream();

    auto kernel_launch = [=]()
        -> detail::task_state
    {
      stream->activate_device();

      // Legacy toolchain does not support __host__ __device__
      // kernel, also we cannot yet enable runtime selection
      // of backends due to name collisions between hipCPU and
      // actual HIP
#ifdef HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
      if(stream->get_device().is_host())
      {
        hipLaunchSequentialKernel(detail::dispatch::host::single_task_kernel,
                                  stream->get_stream(), shared_mem_size,
                                  kernelFunc);
      }
      else
#endif
      {
        __hipsycl_launch_kernel(detail::dispatch::device::single_task_kernel<KernelName>,
                                1,1,shared_mem_size,stream->get_stream(),
                                kernelFunc);
      }

      return detail::task_state::enqueued;
    };

    this->submit_task(kernel_launch);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc)
  {
    dispatch_kernel_without_offset<KernelName>(numWorkItems, kernelFunc);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, KernelType kernelFunc)
  {
    dispatch_kernel_with_offset<KernelName>(numWorkItems, workItemOffset, kernelFunc);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernelFunc)
  {
    dispatch_ndrange_kernel<KernelName>(executionRange, kernelFunc);
  }


  // Hierarchical kernel dispatch API

  /// \todo flexible ranges are currently unsupported
  /*
  template <typename KernelName= class _unnamed_kernel,
            typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               WorkgroupFunctionType kernelFunc)
  {
    dispatch_hierarchical_kernel(numWorkGroups,
                                 get_default_local_range<dimensions>(),
                                 kernelFunc);
  }
  */

  template <typename KernelName = class _unnamed_kernel,
            typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               range<dimensions> workGroupSize,
                               WorkgroupFunctionType kernelFunc)
  {
    dispatch_hierarchical_kernel<KernelName>(numWorkGroups,
                                 workGroupSize,
                                 kernelFunc);
  }


  /*
  void single_task(kernel syclKernel);

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel);

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, kernel syclKernel);

  template <int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel);
  */

  //------ Explicit copy operations API


  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(accessor<T, dim, mode, tgt> src, shared_ptr_class<T> dest)
  {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(shared_ptr_class<T> src, accessor<T, dim, mode, tgt> dest)
  {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(accessor<T, dim, mode, tgt> src, T * dest)
  {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(const T * src, accessor<T, dim, mode, tgt> dest)
  {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode srcMode, access::mode dstMode,
            access::target srcTgt, access::target destTgt>
  void copy(accessor<T, dim, srcMode, srcTgt> src,
            accessor<T, dim, dstMode, destTgt> dest)
  {
    using namespace detail;
    validate_copy_src_accessor(src);
    validate_copy_dest_accessor(dest);

    for(int i = 0; i < dim; ++i)
    {
      if(src.get_range().get(i) > dest.get_range().get(i))
      {
        throw invalid_parameter_error{"sycl explicit copy operation: "
          "Accessor sizes are incompatible."};
      }
    }

    const auto src_ptr = src.get_pointer();
    const auto src_ptr_offset = detail::accessor::get_pointer_offset(src);
    const auto src_buffer_range = detail::accessor::get_buffer_range(src);
    const auto src_acc_range = src.get_range();

    auto dest_ptr = dest.get_pointer();
    const auto dest_ptr_offset = detail::accessor::get_pointer_offset(dest);
    const auto dest_buffer_range = detail::accessor::get_buffer_range(dest);

    constexpr auto copy_kind = get_copy_kind<srcTgt, destTgt>();
    task_graph_node_ptr graph_node = nullptr;

    if(dim == 1) graph_node = dispatch_copy_1d(dest_ptr, dest_ptr_offset, src_ptr,
      src_ptr_offset, detail::range::range_cast<1>(src_acc_range), copy_kind);

    if(dim == 2) graph_node = dispatch_copy_2d(dest_ptr, dest_ptr_offset,
      dest_buffer_range[1], src_ptr, src_ptr_offset, src_buffer_range[1],
      detail::range::range_cast<2>(src_acc_range), copy_kind);

    if(dim == 3) graph_node = dispatch_copy_3d(dest_ptr, dest_ptr_offset,
      detail::range::range_cast<3>(dest_buffer_range), src_ptr, src_ptr_offset,
      detail::range::range_cast<3>(src_buffer_range),
      detail::range::range_cast<3>(src_acc_range), copy_kind);

    maybe_register_copy_access(src, graph_node);
    maybe_register_copy_access(dest, graph_node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void update_host(accessor<T, dim, mode, tgt> acc)
  {
    detail::buffer_ptr buff = _accessor_buffer_map.find_accessor(acc);

    detail::stream_ptr stream = this->get_stream();

    HIPSYCL_DEBUG_INFO << "handler: Spawning async host access task"
                       << std::endl;

    auto task_graph_node = detail::buffer_impl::access_host(
          buff,
          mode,
          stream,
          stream->get_error_handler());

    this->add_access(buff, mode, task_graph_node);
  }

  /// \todo fill() on host accessors can be optimized to use
  /// memset() if the accessor describes a large area of
  /// contiguous memory
  template<typename T, int dim, access::mode mode, access::target tgt>
  void fill(accessor<T, dim, mode, tgt> dest, const T& src)
  {
    static_assert(mode != access::mode::read,
                  "Filling read-only accessors is not allowed.");
    static_assert(tgt != access::target::host_image,
                  "host_image targets are unsupported");

    if(tgt == access::target::host_buffer)
    {
      this->execute_host_range_iteration(dest.get_range(),
                                         dest.get_offset(),
                                         [&](sycl::id<dim> tid){
        dest[tid] = src;
      });
    }
    else
    {
      // Use a function object instead of lambda to avoid
      // requiring a unique kernel name for each fill call
      class fill_kernel
      {
      public:
        fill_kernel(accessor<T, dim, mode, tgt> dest,
                    const T& src)
        : _dest{dest}, _src{src}
        {}

        #ifdef __HIPSYCL_TRANSFORM__
        __device__
        #endif
        void operator()(sycl::id<dim> tid)
        {
          _dest[tid] = _src;
        }

      private:
        accessor<T, dim, mode, tgt> _dest;
        T _src;
      };

      this->parallel_for(
            dest.get_range(),
            dest.get_offset(),
            fill_kernel{dest, src});
    }
  }

  detail::local_memory_allocator& get_local_memory_allocator()
  {
    return _local_mem_allocator;
  }


  event _detail_get_event() const
  {
    if(_spawned_task_nodes.empty())
      return event{};

    return event{_spawned_task_nodes.back()};
  }


  detail::stream_ptr get_stream() const;
private:
  void add_access(detail::buffer_ptr buff,
                  access::mode access_mode,
                  detail::task_graph_node_ptr task)
  {
    this->_spawned_task_nodes.push_back(task);
    this->_accessed_buffers.push_back({access_mode, buff, task});
  }

  struct buffer_access
  {
    access::mode access_mode;
    detail::buffer_ptr buff;
    detail::task_graph_node_ptr task;
  };

  hipStream_t get_hip_stream() const;

  void select_device() const;

  template<int dimensions>
  dim3 get_default_local_range() const
  {
    if(dimensions == 1)
      return dim3(128);
    else if(dimensions == 2)
      return dim3(16,16);
    else if(dimensions == 3)
      return dim3(8,8,8);

    return dim3(1);
  }

  std::size_t ceil_division(std::size_t n,
                           std::size_t divisor) const
  {
    return (n + divisor - 1) / divisor;
  }

  template<int dimensions>
  dim3 range_to_dim3(const range<dimensions>& r) const
  {
    if(dimensions == 1)
      return dim3(r.get(0));
    else if(dimensions == 2)
      return dim3(r.get(0), r.get(1));
    else if(dimensions == 3)
      return dim3(r.get(0), r.get(1), r.get(2));

    return dim3(1);
  }

  template<int dimensions>
  void determine_grid_configuration(const range<dimensions>& num_work_items,
                                    dim3& grid,
                                    dim3& block) const
  {
    block = get_default_local_range<dimensions>();

    if(dimensions == 1)
      grid = dim3(ceil_division(num_work_items.get(0), block.x));
    else if (dimensions == 2)
      grid = dim3(ceil_division(num_work_items.get(0), block.x),
                  ceil_division(num_work_items.get(1), block.y));
    else if (dimensions == 3)
      grid = dim3(ceil_division(num_work_items.get(0), block.x),
                  ceil_division(num_work_items.get(1), block.y),
                  ceil_division(num_work_items.get(2), block.z));
    else
      grid = dim3(1);
  }

  template<typename KernelType, int dimension>
  void execute_host_range_iteration(range<dimension> numWorkItems,
                                    id<dimension> offset,
                                    KernelType f)
  {
    range<3> num_items3d;
    id<3> offset3d;
    for(int i = 0; i < dimension; ++i)
    {
      num_items3d[i] = numWorkItems[i];
      offset3d[i] = offset[i];
    }
    for(int i = dimension; i < 3; ++i)
    {
      num_items3d[i] = 1;
      offset3d[i] = 0;
    }

    id<3> end3d = offset3d + num_items3d;
    id<3> current3d;
    for(current3d[0] = offset3d[0]; current3d[0] < end3d[0]; ++current3d[0])
      for(current3d[1] = offset3d[1]; current3d[1] < end3d[1]; ++current3d[1])
        for(current3d[2] = offset3d[2]; current3d[2] < end3d[2]; ++current3d[2])
        {
          id<dimension> current_item;
          for(int i = 0; i < dimension; ++i)
            current_item[i] = current3d[i];

          f(current_item);
        }
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __host__
  void dispatch_kernel_without_offset(range<dimensions> numWorkItems,
                                      KernelType kernelFunc)
  {
    dim3 grid, block;
    determine_grid_configuration(numWorkItems, grid, block);

    // TODO If shared_mem_size != 0, we can raise an error -
    // local memory doesn't make sense for simple parallel_for!
    std::size_t shared_mem_size =
        _local_mem_allocator.get_allocation_size();

    detail::stream_ptr stream = this->get_stream();

    auto kernel_launch = [=]()
        -> detail::task_state
    {
      stream->activate_device();

#ifdef HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
      if(stream->get_device().is_host())
      {
        hipLaunchSequentialKernel(detail::dispatch::host::parallel_for_kernel,
                                  stream->get_stream(), shared_mem_size,
                                  kernelFunc, numWorkItems, 
                                  detail::dispatch::host::offset::without_offset{});
      }
      else
#endif
      {
        __hipsycl_launch_kernel(detail::dispatch::device::parallel_for_kernel<KernelName>,
                                detail::make_kernel_launch_range<dimensions>(grid),
                                detail::make_kernel_launch_range<dimensions>(block),
                                shared_mem_size, stream->get_stream(),
                                kernelFunc, numWorkItems);
      }

      return detail::task_state::enqueued;
    };

    this->submit_task(kernel_launch);

  }

  template <typename KernelName, typename KernelType, int dimensions>
  __host__
  void dispatch_kernel_with_offset(range<dimensions> numWorkItems,
                                   id<dimensions> offset,
                                   KernelType kernelFunc)
  {
    dim3 grid, block;
    determine_grid_configuration(numWorkItems, grid, block);

    // TODO If shared_mem_size != 0, we can raise an error -
    // local memory doesn't make sense for single_task!
    std::size_t shared_mem_size =
        _local_mem_allocator.get_allocation_size();

    detail::stream_ptr stream = this->get_stream();


    auto kernel_launch = [=]()
        -> detail::task_state
    {
      stream->activate_device();

#ifdef HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
      if(stream->get_device().is_host())
      {
        hipLaunchSequentialKernel(detail::dispatch::host::parallel_for_kernel,
                                  stream->get_stream(), shared_mem_size,
                                  kernelFunc, numWorkItems, 
                                  detail::dispatch::host::offset::with_offset{}, 
                                  offset);
      }
      else
#endif
      {
        __hipsycl_launch_kernel(detail::dispatch::device::parallel_for_kernel_with_offset<KernelName>,
                                detail::make_kernel_launch_range<dimensions>(grid),
                                detail::make_kernel_launch_range<dimensions>(block),
                                shared_mem_size, stream->get_stream(),
                                kernelFunc, numWorkItems, offset);
      }

      return detail::task_state::enqueued;
    };

    this->submit_task(kernel_launch);
  }


  template <typename KernelName, typename KernelType, int dimensions>
  __host__
  void dispatch_ndrange_kernel(nd_range<dimensions> executionRange, KernelType kernelFunc)
  {
    for(int i = 0; i < dimensions; ++i)
    {
      if(executionRange.get_global()[i] % executionRange.get_local()[i] != 0)
        throw invalid_parameter_error{"Global size must be a multiple of the local size"};
    }

    id<dimensions> offset = executionRange.get_offset();
    range<dimensions> grid_range = executionRange.get_group();
    range<dimensions> block_range = executionRange.get_local();

    dim3 grid = range_to_dim3(grid_range);
    dim3 block = range_to_dim3(block_range);

    std::size_t shared_mem_size =
        _local_mem_allocator.get_allocation_size();

    detail::stream_ptr stream = this->get_stream();

    auto kernel_launch = [=]()
        -> detail::task_state
    {
      stream->activate_device();

#ifdef HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
      if(stream->get_device().is_host())
      {
        // We still need the hipCPU kernel launch semantics
        // for ndrange kernels until we have support in the clang
        // plugin for dealing with barriers
        __hipsycl_launch_kernel(detail::dispatch::host::parallel_for_ndrange_kernel,
                                detail::make_kernel_launch_range<dimensions>(grid),
                                detail::make_kernel_launch_range<dimensions>(block),
                                shared_mem_size, stream->get_stream(),
                                kernelFunc, offset);
      }
      else
#endif
      {
        __hipsycl_launch_kernel(detail::dispatch::device::parallel_for_ndrange_kernel<KernelName>,
                                detail::make_kernel_launch_range<dimensions>(grid),
                                detail::make_kernel_launch_range<dimensions>(block),
                                shared_mem_size, stream->get_stream(),
                                kernelFunc, offset);
      }

      return detail::task_state::enqueued;
    };

    this->submit_task(kernel_launch);
  }

  template <typename KernelName, typename WorkgroupFunctionType, int dimensions>
  __host__
  void dispatch_hierarchical_kernel(range<dimensions> numWorkGroups,
                                    range<dimensions> workGroupSize,
                                    WorkgroupFunctionType kernelFunc)
  {

    std::size_t shared_mem_size =
        _local_mem_allocator.get_allocation_size();

    detail::stream_ptr stream = this->get_stream();

    dim3 grid = range_to_dim3(numWorkGroups);
    dim3 block = range_to_dim3(workGroupSize);

    auto kernel_launch = [=]()
        -> detail::task_state
    {
      stream->activate_device();

#ifdef HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
      if(stream->get_device().is_host())
      {
        hipLaunchSequentialKernel(detail::dispatch::host::parallel_for_workgroup,
                                  stream->get_stream(), 0,
                                  kernelFunc, numWorkGroups, workGroupSize, 
                                  shared_mem_size);
      }
      else
#endif
      {
        __hipsycl_launch_kernel(detail::dispatch::device::parallel_for_workgroup<KernelName>,
                                detail::make_kernel_launch_range<dimensions>(grid),
                                detail::make_kernel_launch_range<dimensions>(block),
                                shared_mem_size, stream->get_stream(),
                                kernelFunc, workGroupSize);
      }


      return detail::task_state::enqueued;
    };

    this->submit_task(kernel_launch);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            typename destPtr>
  void copy_ptr(accessor<T, dim, mode, tgt> src, destPtr dest)
  {
    validate_copy_src_accessor(src);

    const auto src_ptr = src.get_pointer();
    const auto src_ptr_offset = detail::accessor::get_pointer_offset(src);
    const auto src_buffer_range = detail::accessor::get_buffer_range(src);
    const auto src_acc_range = src.get_range();

    constexpr auto copy_kind = detail::get_copy_kind<tgt, access::target::host_buffer>();
    detail::task_graph_node_ptr graph_node = nullptr;

    if(dim == 1) graph_node = dispatch_copy_1d(dest, 0, src_ptr, src_ptr_offset,
      detail::range::range_cast<1>(src_acc_range), copy_kind);

    if(dim == 2) graph_node = dispatch_copy_2d(dest, 0, src_acc_range[1], src_ptr,
      src_ptr_offset, src_buffer_range[1],
      detail::range::range_cast<2>(src_acc_range), copy_kind);

    if(dim == 3) graph_node = dispatch_copy_3d(dest, 0,
      detail::range::range_cast<3>(src_acc_range),
      src_ptr, src_ptr_offset, detail::range::range_cast<3>(src_buffer_range),
      detail::range::range_cast<3>(src_acc_range), copy_kind);

    maybe_register_copy_access(src, graph_node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            typename srcPtr>
  void copy_ptr(srcPtr src, accessor<T, dim, mode, tgt> dest)
  {
    validate_copy_dest_accessor(dest);

    auto dest_ptr = dest.get_pointer();
    const auto dest_ptr_offset = detail::accessor::get_pointer_offset(dest);
    const auto dest_buffer_range = detail::accessor::get_buffer_range(dest);
    const auto dest_acc_range = dest.get_range();

    constexpr auto copy_kind = detail::get_copy_kind<access::target::host_buffer, tgt>();
    detail::task_graph_node_ptr graph_node = nullptr;

    if(dim == 1) graph_node = dispatch_copy_1d(dest_ptr, dest_ptr_offset, src,
      0, detail::range::range_cast<1>(dest_acc_range), copy_kind);

    if(dim == 2) graph_node = dispatch_copy_2d(dest_ptr, dest_ptr_offset,
      dest_buffer_range[1], src, 0, dest_acc_range[1],
      detail::range::range_cast<2>(dest_acc_range), copy_kind);

    if(dim == 3) graph_node = dispatch_copy_3d(dest_ptr, dest_ptr_offset,
      detail::range::range_cast<3>(dest_buffer_range), src, 0,
      detail::range::range_cast<3>(dest_acc_range),
      detail::range::range_cast<3>(dest_acc_range), copy_kind);

    maybe_register_copy_access(dest, graph_node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void validate_copy_src_accessor(const accessor<T, dim, mode, tgt>&)
  {
    static_assert(dim != 0, "0-dimensional accessors are currently not supported");
    static_assert(mode == access::mode::read || mode == access::mode::read_write,
      "Only read or read_write accessors can be copied from");
    static_assert(tgt == access::target::global_buffer ||
      tgt == access::target::host_buffer,
      "Only global_buffer or host_buffer accessors are currently "
      "supported for copying");
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void validate_copy_dest_accessor(const accessor<T, dim, mode, tgt>&)
  {
    static_assert(dim != 0, "0-dimensional accessors are currently not supported");
    static_assert(mode == access::mode::write ||
      mode == access::mode::read_write ||
      mode == access::mode::discard_write ||
      mode == access::mode::discard_read_write,
      "Only write, read_write, discard_write or "
      "discard_read_write accessors can be copied to");
    static_assert(tgt == access::target::global_buffer ||
      tgt == access::target::host_buffer,
      "Only global_buffer or host_buffer accessors are currently "
      "supported for copying");
  }

  void debug_print_copy_kind(hipMemcpyKind kind) {
    switch(kind) {
      case hipMemcpyHostToHost:
        HIPSYCL_DEBUG_INFO <<
          "handler: Spawning async host to host copy task" << std::endl;
        break;
      case hipMemcpyHostToDevice:
        HIPSYCL_DEBUG_INFO <<
          "handler: Spawning async host to device copy task" << std::endl;
        break;
      case hipMemcpyDeviceToHost:
        HIPSYCL_DEBUG_INFO <<
          "handler: Spawning async device to host copy task" << std::endl;
        break;
      case hipMemcpyDeviceToDevice:
        HIPSYCL_DEBUG_INFO <<
          "handler: Spawning async device to device copy task" << std::endl;
        break;
      default: assert(false);
    }
  }

  template <typename destPtr, typename srcPtr>
  detail::task_graph_node_ptr dispatch_copy_1d(destPtr dest, size_t dest_offset,
                                               const srcPtr src, size_t src_offset,
                                               range<1> count, hipMemcpyKind kind)
  {
    using namespace detail;
    using T = std::remove_pointer_t<decltype(get_raw_pointer(src))>;
    debug_print_copy_kind(kind);
    const auto stream = this->get_stream();
    auto copy_launch = [=]() -> task_state
    {
      stream->activate_device();

      detail::check_error(
        hipMemcpyAsync(get_raw_pointer(dest) + dest_offset, get_raw_pointer(src) +
          src_offset, count[0] * sizeof(T), kind, stream->get_stream()));

      return task_state::enqueued;
    };
    return this->submit_task(copy_launch);
  }

  template <typename destPtr, typename srcPtr>
  detail::task_graph_node_ptr dispatch_copy_2d(destPtr dest, size_t dest_offset,
                                               size_t dest_pitch, const srcPtr src,
                                               size_t src_offset, size_t src_pitch,
                                               range<2> count, hipMemcpyKind kind)
  {
    using namespace detail;
    using T = std::remove_pointer_t<decltype(get_raw_pointer(src))>;
    debug_print_copy_kind(kind);
    const auto stream = this->get_stream();
    auto copy_launch = [=]() -> task_state
    {
      stream->activate_device();

      detail::check_error(
        hipMemcpy2DAsync(get_raw_pointer(dest) + dest_offset, dest_pitch * sizeof(T),
          get_raw_pointer(src) + src_offset, src_pitch * sizeof(T),
          count[1] * sizeof(T), count[0], kind, stream->get_stream()));

      return task_state::enqueued;
    };
    return this->submit_task(copy_launch);
  }

  template <typename destPtr, typename srcPtr>
  detail::task_graph_node_ptr dispatch_copy_3d(destPtr dest, size_t dest_offset,
                                               range<3> dest_buffer_range,
                                               const srcPtr src, size_t src_offset,
                                               range<3> src_buffer_range,
                                               range<3> count, hipMemcpyKind kind)
  {
#if defined(HIPSYCL_PLATFORM_CUDA)
    using namespace detail;
    using T = std::remove_pointer_t<decltype(get_raw_pointer(src))>;
    debug_print_copy_kind(kind);
    const auto stream = this->get_stream();

    auto copy_launch = [=]() -> task_state
    {
      hipMemcpy3DParms params = {};
      auto raw_src_ptr = const_cast<std::remove_const_t<T>*>(get_raw_pointer(src));
      params.srcPtr = make_hipPitchedPtr(raw_src_ptr + src_offset,
        src_buffer_range[2] * sizeof(T), src_buffer_range[2], src_buffer_range[1]);
      params.dstPtr = make_hipPitchedPtr( get_raw_pointer(dest) + dest_offset,
        dest_buffer_range[2] * sizeof(T), dest_buffer_range[2], dest_buffer_range[1]);
      params.extent = {count[2] * sizeof(T), count[1], count[0]};
      // hipMemcpy3DParms on CUDA is simply a typedef, so it actually requires a cudaMemcpyKind.
      // Thankfully the enums are compatible.
      params.kind = static_cast<cudaMemcpyKind>(
        static_cast<std::underlying_type_t<hipMemcpyKind>>(kind));

      stream->activate_device();

      auto err = cudaMemcpy3DAsync(&params, stream->get_stream());
      detail::check_error(hipCUDAErrorTohipError(err));

      return task_state::enqueued;
    };
    return this->submit_task(copy_launch);
#else
    // It looks like HIP doesn't provide a hipMemcpy3DAsync as of April 2019.
    // See https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md
    // TODO: We could fall back to hipMemcpy3D on HCC (however, that requires a different parameter struct).
    throw feature_not_supported{"3D copy() is currently not supported on "
      "this platform"};
#endif
  }

  /// Registers an external access for host-accessed buffers, which is required
  /// to ensure that subsequent host accesses wait for explicit copy operations.
  template <typename T, int dim, access::mode mode, access::target tgt>
  void maybe_register_copy_access(accessor<T, dim, mode, tgt>& acc,
                                  detail::task_graph_node_ptr task_node)
  {
    if(tgt != access::target::host_buffer) return;
    
    detail::buffer_ptr buff = _accessor_buffer_map.find_accessor(acc);
    buff->register_external_access(task_node, mode);
    HIPSYCL_DEBUG_INFO << "handler: Registering external access via task "
      << task_node << " for buffer " << buff << std::endl;
  }

  detail::task_graph_node_ptr submit_task(detail::task_functor f)
  {
    auto& task_graph = detail::application::get_task_graph();

    auto graph_node =
        task_graph.insert(f, _spawned_task_nodes, get_stream(), _handler);

    // Add new node to the access log of buffers. This guarantees that
    // subsequent buffer accesses will wait for existing tasks to complete,
    // if necessary
    for(const auto& buffer_access : _accessed_buffers)
    {
      buffer_access.buff->register_external_access(
            graph_node,
            buffer_access.access_mode);
    }

    _spawned_task_nodes.push_back(graph_node);
    return graph_node;
  }

  detail::accessor_id request_accessor_id(detail::buffer_ptr buff)
  {
    return _accessor_buffer_map.insert(buff);
  }

  handler(const queue& q, async_handler handler);

  friend class queue;
  friend void* detail::accessor::obtain_device_access(
    detail::buffer_ptr, sycl::handler&, access::mode);
  friend detail::accessor_id detail::accessor::request_accessor_id(
    detail::buffer_ptr, sycl::handler&);

  const queue* _queue;
  detail::local_memory_allocator _local_mem_allocator;
  async_handler _handler;


  vector_class<detail::task_graph_node_ptr> _spawned_task_nodes;
  vector_class<buffer_access> _accessed_buffers;
  detail::accessor_buffer_mapper _accessor_buffer_map;
};

namespace detail {
namespace handler {


template<class T>
inline detail::local_memory::address allocate_local_mem(
    sycl::handler& cgh,
    size_t num_elements)
{
  return cgh.get_local_memory_allocator().alloc<T>(num_elements);
}

}
}

} // namespace sycl
} // namespace hipsycl

#endif
