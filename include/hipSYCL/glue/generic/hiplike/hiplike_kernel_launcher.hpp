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
#include <utility>
#include <cstdlib>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/application.hpp"

#include "clang.hpp"
#include "hiplike_reducer.hpp"


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

// Invoke on device and construct local_reducer objects
template <class F, typename... Reductions>
__host__ __device__ void
device_invocation_with_local_reducer(F f, Reductions... reductions) {

#ifdef SYCL_DEVICE_ONLY
  auto invoker = [&, f] __host__ __device__ (auto... reducers) {

    f(reducers...);
    (reducers.finalize_result(), ...);
  };
  invoker(reductions.construct_reducer()...);
#else
  assert(false && "Attempted to execute device code path on host!");
#endif
}

template<class F, typename... Reductions>
__host__ __device__
void device_invocation(F f, Reductions... reductions)
{
  device_invocation_with_local_reducer([&](auto& ... local_reducers){
    f(sycl::reducer{local_reducers}...);
  }, reductions...);
}

template <typename KernelName, typename... Reductions>
__sycl_kernel void reduction_kernel(int num_input_elements,
                                Reductions... reductions) {
  device_invocation_with_local_reducer(
      [&] __host__ __device__(auto &... local_reducers) {
        int gid = __hipsycl_lid_x + __hipsycl_gid_x * __hipsycl_lsize_x;
        if(gid < num_input_elements){
          (local_reducers.combine_global_input(gid), ...);
        }
      },
      reductions...);
}

template<typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f)
{
  device_invocation(f);
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void
parallel_for_kernel(Function f, sycl::range<dimensions> execution_range,
                    sycl::id<dimensions> offset, bool with_offset,
                    Reductions... reductions) {
  // Note: We currently cannot have with_offset as template parameter
  // because this might cause clang to emit two kernels with the same
  // mangled name (variants with and without offset) if an explicit kernel
  // name is provided.
  if(with_offset) {
    device_invocation([&] __host__ __device__(auto... reducers) {
      auto this_item = sycl::detail::make_item<dimensions>(
          sycl::detail::get_global_id<dimensions>() + offset, execution_range,
          offset);
      if (item_is_in_range(this_item, execution_range, offset))
        f(this_item, reducers...);
    }, reductions...);
  } else {
    device_invocation([&] __host__ __device__(auto... reducers) {
      auto this_item = sycl::detail::make_item<dimensions>(
          sycl::detail::get_global_id<dimensions>(), execution_range);
      if (item_is_in_range(this_item, execution_range))
        f(this_item, reducers...);
    }, reductions...);
  }
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void parallel_for_ndrange_kernel(Function f,
                                               sycl::id<dimensions> offset,
                                               Reductions... reductions) {
  device_invocation([&] __host__ __device__(auto... reducers) {
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
    f(this_item, reducers...);
  }, reductions...);
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void
parallel_for_workgroup(Function f,
                       // The logical group size is not yet used,
                       // but it's still useful to already have it here
                       // since it allows the compiler to infer 'dimensions'
                       sycl::range<dimensions> logical_group_size,
                       Reductions... reductions) {
  device_invocation([&] __host__ __device__(auto... reducers) {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    sycl::group<dimensions> this_group;
#else
    sycl::group<dimensions> this_group{
        sycl::detail::get_group_id<dimensions>(),
        sycl::detail::get_local_size<dimensions>(),
        sycl::detail::get_grid_size<dimensions>()};
#endif
    f(this_group, reducers...);
  }, reductions...);
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void parallel_region(Function f,
                                   sycl::range<dimensions> num_groups,
                                   sycl::range<dimensions> group_size,
                                   Reductions... reductions) {
  device_invocation([&] __host__ __device__ (auto... reducers) {
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
    
    f(this_group, phys_idx, reducers...);
  }, reductions...);
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


template <int Dimensions> struct reduction_stage {

  reduction_stage(const sycl::range<Dimensions> &local_size,
                  const sycl::range<Dimensions> &num_groups,
                  const sycl::range<Dimensions> &global_size)
      : local_size{local_size}, num_groups{num_groups},
        global_size{global_size}, allocated_local_memory{0} {}

  sycl::range<Dimensions> local_size;
  sycl::range<Dimensions> num_groups;
  sycl::range<Dimensions> global_size;
  int allocated_local_memory;
};

template <int Dimensions>
std::vector<reduction_stage<Dimensions>>
determine_reduction_stages(sycl::range<Dimensions> global_size,
                           sycl::range<Dimensions> local_size,
                           sycl::range<Dimensions> num_groups) {
  std::vector<reduction_stage<Dimensions>> stages;

  auto is_final_reduce = [](sycl::range<Dimensions> num_groups) {
    return num_groups.size() == 1;
  };

  stages.push_back(reduction_stage<Dimensions>{
      local_size, num_groups, global_size});

  sycl::range<Dimensions> current_num_groups = num_groups;
  sycl::range<Dimensions> current_num_work_items = global_size;

  while(current_num_groups.size() > 1) {

    current_num_work_items = current_num_groups;
    current_num_groups =
      determine_grid_configuration(current_num_groups, local_size);

    // ToDo: Might want to use a different local size for pure
    // reduce steps
    stages.push_back(reduction_stage<Dimensions>{
        local_size, current_num_groups, current_num_work_items});
  }
  return stages;
}


template<class ReductionDescriptor, class ReductionStage>
class hiplike_reduction_descriptor {
public:
  using value_type = typename ReductionDescriptor::value_type;

  __host__ hiplike_reduction_descriptor(
      const ReductionDescriptor& desc,
      rt::device_id dev, const std::vector<ReductionStage> &stages,
      std::vector<void *> &managed_scratch_memory)
      : _descriptor{desc}, _is_final{false}, _scratch_memory_in{nullptr},
        _scratch_memory_out{nullptr} {
    
    assert(stages.size() > 0);

    std::size_t num_scratch_elements = stages[0].num_groups.size();
    std::size_t scratch_memory_size =
        sizeof(value_type) * num_scratch_elements;

    auto allocator =
        rt::application::get_backend(dev.get_backend()).get_allocator(dev);

    auto allocate_scratch = [&]() -> void * {
      void *mem = allocator->allocate(0, scratch_memory_size);

      if (!mem) {
        rt::register_error(
            __hipsycl_here(),
            rt::error_info{"Could not allocate scratch memory for reduction",
                           rt::error_type::memory_allocation_error});
        return nullptr;
      }

      managed_scratch_memory.push_back(mem);
      return mem;
    };

    if (stages.size() > 1) {
      this->_scratch_memory_out = allocate_scratch();
      if (stages.size() > 2) {
        this->_scratch_memory_in = allocate_scratch();
      }
    }
  }

  __host__ void proceed_to_stage(std::size_t stage_index,
                                 std::size_t num_stages,
                                 ReductionStage &stage) {

    this->initialize_local_memory(stage.local_size.size(),
                                  stage.allocated_local_memory);

    if (stage_index + 1 == num_stages)
      _is_final = true;
  
    if (stage_index > 0) {
      std::swap(_scratch_memory_in, _scratch_memory_out);
    }
  }

  // Must be __host__ __device__ in order to be able to call
  // hiplike_dynamic_local_memory()
  __host__ __device__
  void* get_local_scratch_mem() const {
    return static_cast<void *>(
        static_cast<char *>(sycl::detail::hiplike_dynamic_local_memory()) +
        _local_memory_offset);
  }

  __host__ __device__ 
  void* get_reduction_output_buffer() const {
#ifdef SYCL_DEVICE_ONLY
    if(!_is_final)
      return _scratch_memory_out;
    else
      return _descriptor.get_pointer();
#else
    return nullptr;
#endif
  }

  __device__ hiplike::local_reducer<ReductionDescriptor>
  construct_reducer() const {
    int my_local_id = __hipsycl_lid_z * __hipsycl_lsize_y * __hipsycl_lsize_x +
                      __hipsycl_lid_y * __hipsycl_lsize_x + __hipsycl_lid_x;
    int my_group_id =
        __hipsycl_gid_z * __hipsycl_ngroups_y * __hipsycl_ngroups_x +
        __hipsycl_gid_y * __hipsycl_ngroups_x + __hipsycl_gid_x;

    value_type *group_output_ptr =
        static_cast<value_type *>(get_reduction_output_buffer()) + my_group_id;

    value_type* global_input_ptr =
        static_cast<value_type*>(get_reduction_input_buffer());

    return hiplike::local_reducer<ReductionDescriptor>{
        _descriptor, my_local_id,
        static_cast<value_type *>(get_local_scratch_mem()), group_output_ptr,
        global_input_ptr};
  }

  __device__ void *get_reduction_input_buffer() const {
    return _scratch_memory_in;
  }

  __host__ __device__
  const ReductionDescriptor& get_descriptor() const {
    return _descriptor;
  }

  
private:
  
  __host__ void initialize_local_memory(int work_group_size,
                                        int &allocated_local_mem_size) {
    
    std::size_t alignment = alignof(value_type);
    std::size_t element_size = sizeof(value_type);

    this->_local_memory_offset =
        ceil_division(allocated_local_mem_size, alignment) *
        alignment;

    allocated_local_mem_size = _local_memory_offset + 
        work_group_size * sizeof(value_type);
  }

  bool _is_final;
  
  void* _scratch_memory_in;
  void* _scratch_memory_out;

  int _local_memory_offset;
  ReductionDescriptor _descriptor;
};

template <class F, class ReductionStage, typename... ReductionDescriptors>
__host__
void invoke_reducible_kernel(F &&handler, rt::device_id dev,
                              const std::vector<ReductionStage>& stages,
                              std::vector<void *> &managed_scratch_memory,
                              ReductionDescriptors&& ... descriptors) {
  handler(hiplike_reduction_descriptor{descriptors, dev, stages,
                                       managed_scratch_memory}...);
}

} // hiplike_dispatch

template<rt::backend_id Backend_id, class Queue_type>
class hiplike_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  hiplike_kernel_launcher()
      : _queue{nullptr}, _invoker{[]() {}} {}

  virtual ~hiplike_kernel_launcher() {
    assert(_queue);
    for(void* scratch_ptr : _managed_reduction_scratch) {
      rt::application::get_backend(Backend_id)
          .get_allocator(_queue->get_device())
          ->free(scratch_ptr);
    }
  }

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

    sycl::range<Dim> effective_local_range = local_range;
    if constexpr (type == rt::kernel_type::basic_parallel_for) {
      if constexpr (Dim == 1)
        effective_local_range = sycl::range<1>{128};
      else if constexpr (Dim == 2)
        effective_local_range = sycl::range<2>{16, 16};
      else if constexpr (Dim == 3)
        effective_local_range = sycl::range<3>{4, 8, 8};
    }

    sycl::range<Dim> grid_range =
        hiplike_dispatch::determine_grid_configuration(global_range,
                                                       effective_local_range);

    _invoker = [=]() {
      assert(_queue != nullptr);

      std::vector<hiplike_dispatch::reduction_stage<Dim>> reduction_stages;
      constexpr bool has_reductions = sizeof...(Reductions) > 0;

      if constexpr (has_reductions) {
        reduction_stages = hiplike_dispatch::determine_reduction_stages(
            global_range, local_range, grid_range);
      }

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      if constexpr (type == rt::kernel_type::single_task) {
       
        __hipsycl_launch_kernel(
            hiplike_dispatch::single_task_kernel<KernelName>, 1, 1,
            dynamic_local_memory, _queue->get_stream(), k);

      } else if constexpr (type == rt::kernel_type::custom) {
       
        sycl::interop_handle handle{_queue->get_device(),
                                    static_cast<void *>(_queue)};

        // Need to perform additional copy to guarantee
        // deferred_pointers/ accessors are initialized
        auto initialized_kernel_invoker = k;
        initialized_kernel_invoker(handle);
      }
      else {

        auto reducible_kernel_invoker =
            [&] (auto... reduction_descriptors) {
              for (std::size_t stage = 0; stage < reduction_stages.size();
                   ++stage) {

                (reduction_descriptors.proceed_to_stage(
                     stage, reduction_stages.size(),
                     reduction_stages[stage]),
                 ...);

                // Reductions will need local memory only *after* the
                // user-provided
                // kernel has completed, so we can reuse the same memory
                int required_dynamic_local_mem = std::max(
                    static_cast<int>(dynamic_local_memory),
                    reduction_stages[stage].allocated_local_memory);
                
                if (stage == 0) {
                  if constexpr (type == rt::kernel_type::basic_parallel_for) {

                    __hipsycl_launch_kernel(
                        hiplike_dispatch::parallel_for_kernel<KernelName>,
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            grid_range),
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            effective_local_range),
                        required_dynamic_local_mem, _queue->get_stream(), k,
                        global_range, offset, is_with_offset,
                        reduction_descriptors...);

                  } else if constexpr (type ==
                                       rt::kernel_type::ndrange_parallel_for) {

                    for (int i = 0; i < Dim; ++i)
                      assert(global_range[i] % effective_local_range[i] == 0);

                    __hipsycl_launch_kernel(
                        hiplike_dispatch::parallel_for_ndrange_kernel<
                            KernelName>,
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            grid_range),
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            effective_local_range),
                        required_dynamic_local_mem, _queue->get_stream(), k,
                        offset, reduction_descriptors...);

                  } else if constexpr (type == rt::kernel_type::
                                                   hierarchical_parallel_for) {

                    for (int i = 0; i < Dim; ++i)
                      assert(global_range[i] % effective_local_range[i] == 0);

                    __hipsycl_launch_kernel(
                        hiplike_dispatch::parallel_for_workgroup<KernelName>,
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            grid_range),
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            effective_local_range),
                        required_dynamic_local_mem, _queue->get_stream(), k,
                        effective_local_range, reduction_descriptors...);

                  } else if constexpr (type ==
                                       rt::kernel_type::scoped_parallel_for) {

                    for (int i = 0; i < Dim; ++i)
                      assert(global_range[i] % effective_local_range[i] == 0);

                    __hipsycl_launch_kernel(
                        hiplike_dispatch::parallel_region<KernelName>,
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            grid_range),
                        hiplike_dispatch::make_kernel_launch_range<Dim>(
                            effective_local_range),
                        required_dynamic_local_mem, _queue->get_stream(), k,
                        grid_range, effective_local_range,
                        reduction_descriptors...);

                  } else {
                    assert(false && "Unsupported kernel type");
                  }
            } else {
              // Launch dedicated reduction kernel
              // 
              // Avoid instantiation of dedicated reduction kernel
              // if there are no reductions for compatibility reasons
              // (requires unique lambda name mangling if the reduction
              //  combiner is a lambda function)
              if constexpr (has_reductions) {
                std::size_t num_groups = reduction_stages[stage].num_groups.size();
                std::size_t local_size = reduction_stages[stage].local_size.size();

                __hipsycl_launch_kernel(
                    hiplike_dispatch::reduction_kernel<class _force_unnamed_kernel>,
                    hiplike_dispatch::make_kernel_launch_range<1>(sycl::range<1>{num_groups}),
                    hiplike_dispatch::make_kernel_launch_range<1>(sycl::range<1>{local_size}),
                    reduction_stages[stage].allocated_local_memory,
                    _queue->get_stream(),
                    reduction_stages[stage].global_size.size(),
                    reduction_descriptors...);
              }
            }
          }
        };
     
        hiplike_dispatch::invoke_reducible_kernel(
            reducible_kernel_invoker, _queue->get_device(), reduction_stages,
            _managed_reduction_scratch, reductions...);
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
  
private:
  Queue_type *_queue;
  rt::kernel_type _type;
  std::function<void ()> _invoker;

  std::vector<void*> _managed_reduction_scratch;
};

}
}

#endif
