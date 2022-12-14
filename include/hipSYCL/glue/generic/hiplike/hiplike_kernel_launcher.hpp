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
#include <string>

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                              \
    HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP
 #define HIPSYCL_HIPLIKE_LAUNCHER_ALLOW_DEVICE_CODE
#endif


#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"

#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/dag_node.hpp"

#include "hipSYCL/glue/kernel_names.hpp"
#include "hipSYCL/glue/generic/code_object.hpp"


#if defined(HIPSYCL_HIPLIKE_LAUNCHER_ALLOW_DEVICE_CODE)

 #if !defined(HIPSYCL_LIBKERNEL_CUDA_NVCXX)
  #include "clang.hpp"
 #else
  #include "nvcxx.hpp"
 #endif
 #include "hiplike_reducer.hpp"
 #include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
 
#else

#ifndef __host__
 #define __host__
#endif
#ifndef __device__
 #define __device__
#endif
#ifndef __global__
 #define __global__
#endif
#ifndef __sycl_kernel
 #define __sycl_kernel
#endif
#ifndef __hipsycl_launch_integrated_kernel

#define __hipsycl_launch_integrated_kernel(f, grid, block, shared_mem, stream, \
                                           ...)                                \
  assert(false && "Dummy integrated kernel launch was called");

struct dim3 {
  dim3() = default;
  dim3(unsigned int x_, unsigned int y_, unsigned int z_)
  : x{x_}, y{y_}, z{z_} {}

  unsigned int x;
  unsigned int y;
  unsigned int z;
};

#endif
#endif

// Dummy kernel that is used in conjunction with 
// __builtin_get_device_side_stable_name() to extract kernel names
template<class KernelT>
__global__ void __hipsycl_kernel_name_template () {}

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
device_invocation_with_local_reducers(F&& f, Reductions... reductions) {

  __hipsycl_if_target_device(
    auto invoker = [&, f] __host__ __device__ (auto... reducers) {

      f(reducers...);
      (reducers.finalize_result(), ...);
    };
    invoker(reductions.construct_reducer()...);
  );
  __hipsycl_if_target_host(
    assert(false && "Attempted to execute device code path on host!");
  );
}

template<class F, typename... Reductions>
__host__ __device__
void device_invocation(F&& f, Reductions... reductions)
{
  device_invocation_with_local_reducers([&](auto& ... local_reducers){
    f(sycl::reducer{local_reducers}...);
  }, reductions...);
}

template <typename KernelName, class Function, typename... Reductions>
__sycl_kernel void
primitive_parallel_for_with_local_reducers(Function f,
                                           Reductions... reductions) {
  __hipsycl_if_target_device(
    device_invocation_with_local_reducers(
        [&] __host__ __device__(auto &...local_reducers) {
          int gid = __hipsycl_lid_x + __hipsycl_gid_x * __hipsycl_lsize_x;
          f(gid, local_reducers...);
        },
        reductions...);
  );
}

template <typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f) {
  __hipsycl_if_target_device(
    device_invocation(f);
  );
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void
parallel_for_kernel(Function f, sycl::range<dimensions> execution_range,
                    sycl::id<dimensions> offset, bool with_offset,
                    Reductions... reductions) {
  __hipsycl_if_target_device(
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
  );
}

template <typename KernelName, class Function, int dimensions,
          typename... Reductions>
__sycl_kernel void parallel_for_ndrange_kernel(Function f,
                                               sycl::id<dimensions> offset,
                                               Reductions... reductions) {
  __hipsycl_if_target_device(
    device_invocation(
      [&] __host__ __device__(auto... reducers) {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        sycl::nd_item<dimensions> this_item{&offset};
#else
        sycl::nd_item<dimensions> this_item{
            &offset, sycl::detail::get_group_id<dimensions>(),
            sycl::detail::get_local_id<dimensions>(),
            sycl::detail::get_local_size<dimensions>(),
            sycl::detail::get_grid_size<dimensions>()};
#endif
        f(this_item, reducers...);
      },
      reductions...);
  );
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
  __hipsycl_if_target_device(
    device_invocation(
        [&] __host__ __device__(auto... reducers) {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          sycl::group<dimensions> this_group;
#else
          sycl::group<dimensions> this_group{
              sycl::detail::get_group_id<dimensions>(),
              sycl::detail::get_local_size<dimensions>(),
              sycl::detail::get_grid_size<dimensions>()};
#endif
          f(this_group, reducers...);
        },
        reductions...);
  );
}

template<int DivisorX = 1, int DivisorY = 1, int DivisorZ = 1>
struct sp_multiversioning_properties {
  static constexpr int group_divisor_x = DivisorX;
  static constexpr int group_divisor_y = DivisorY;
  static constexpr int group_divisor_z = DivisorZ;

  template<int Dim>
  static constexpr auto get_sp_property_descriptor() {
    return sycl::detail::sp_property_descriptor<
        Dim, 0, decltype(get_hierarchical_decomposition<Dim>())>{};
  }

private:
  template<int Dim>
  static constexpr auto get_hierarchical_decomposition() {
    using namespace sycl::detail;
    using fallback_decomposition =
      nested_range<unknown_static_range, nested_range<static_range<1>>>;

    if constexpr(Dim == 1) {
      if constexpr(DivisorX % __hipsycl_warp_size == 0) {
        using decomposition =
            nested_range<unknown_static_range,
                         nested_range<static_range<__hipsycl_warp_size>>>;

        return decomposition{};
      } else {
        return fallback_decomposition{};
      }
    } else if constexpr(Dim == 2) {
      return fallback_decomposition{};
    } else {
      return fallback_decomposition{};
    }
  }
};

template <typename KernelName, class Function, class MultiversioningProps,
          int dimensions, typename... Reductions>
__sycl_kernel void
parallel_region(Function f, MultiversioningProps props,
                sycl::range<dimensions> num_groups,
                sycl::range<dimensions> group_size, Reductions... reductions) {
  __hipsycl_if_target_device(
    device_invocation(
      [&] __host__ __device__(auto... reducers) {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        sycl::group<dimensions> this_group;
#else
        sycl::group<dimensions> this_group{
            sycl::detail::get_group_id<dimensions>(),
            sycl::detail::get_local_size<dimensions>(),
            sycl::detail::get_grid_size<dimensions>()};
#endif
        using group_properties = std::decay_t<
            decltype(MultiversioningProps::template get_sp_property_descriptor<
                     dimensions>())>;
        f(sycl::detail::sp_group<group_properties>{this_group}, reducers...);
      },
      reductions...);
  );
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

  // add user-provided, initial stage
  stages.push_back(reduction_stage<Dimensions>{
      local_size, num_groups, global_size});

  // Reduce ranges of the form (N,M,L) to (1,1,N*M*L)
  global_size[Dimensions - 1] = global_size.size();
  for (int i=1; i<Dimensions; ++i)
    global_size[i] = 1;

  local_size[Dimensions - 1] = local_size.size();
  for (int i=1; i<Dimensions; ++i)
    local_size[i] = 1;
  // Check if local_size == 1, this would cause an infinite loop
  if (local_size[0] == 1)
    local_size[0] = 128;

  num_groups[Dimensions - 1] = num_groups.size();
  for (int i=1; i<Dimensions; ++i)
    num_groups[i] = 1;

  sycl::range<Dimensions> current_num_groups = num_groups;
  sycl::range<Dimensions> current_num_work_items = global_size;
  
  while(current_num_groups[Dimensions - 1] > 1) {

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
      const std::vector<ReductionStage> &stages,
      std::vector<void *> &managed_scratch_memory,
      rt::backend_allocator* allocator)
      : _descriptor{desc}, _is_final{false}, _scratch_memory_in{nullptr},
        _scratch_memory_out{nullptr}, _allocator{allocator} {
    
    assert(stages.size() > 0);

    std::size_t num_scratch_elements = stages[0].num_groups.size();
    std::size_t scratch_memory_size =
        sizeof(value_type) * num_scratch_elements;

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
  __host__ __device__ void *get_local_scratch_mem() const {
    __hipsycl_if_target_device(
        return static_cast<void *>(
                   static_cast<char *>(
                       sycl::detail::hiplike_dynamic_local_memory()) +
                   _local_memory_offset);
    );
    __hipsycl_if_target_host(
      return nullptr;
    );
  }

  __host__ __device__ 
  void* get_reduction_output_buffer() const {
    __hipsycl_if_target_device(
      if(!_is_final)
        return _scratch_memory_out;
      else
        return _descriptor.get_pointer();
    );
    __hipsycl_if_target_host(
      return nullptr;
    );
  }


  __device__ hiplike::local_reducer<ReductionDescriptor>
  construct_reducer() const {
    __hipsycl_if_target_device(
        int my_local_id =
            __hipsycl_lid_z * __hipsycl_lsize_y * __hipsycl_lsize_x +
            __hipsycl_lid_y * __hipsycl_lsize_x + __hipsycl_lid_x;
        int my_group_id =
            __hipsycl_gid_z * __hipsycl_ngroups_y * __hipsycl_ngroups_x +
            __hipsycl_gid_y * __hipsycl_ngroups_x + __hipsycl_gid_x;

        value_type *group_output_ptr =
            static_cast<value_type *>(get_reduction_output_buffer()) +
            my_group_id;

        value_type *global_input_ptr =
            static_cast<value_type *>(get_reduction_input_buffer());

        return hiplike::local_reducer<ReductionDescriptor>{
            _descriptor, my_local_id,
            static_cast<value_type *>(get_local_scratch_mem()),
            group_output_ptr, global_input_ptr};
    );
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
  rt::backend_allocator* _allocator;
};

template <class F, class ReductionStage, typename... ReductionDescriptors>
__host__ void
invoke_reducible_kernel(F &&handler, const std::vector<ReductionStage> &stages,
                        std::vector<void *> &managed_scratch_memory,
                        rt::backend_allocator *allocator,
                        ReductionDescriptors &&...descriptors) {
  handler(hiplike_reduction_descriptor{descriptors, stages,
                                       managed_scratch_memory, allocator}...);
}

} // hiplike_dispatch

template<rt::backend_id Backend_id, class Queue_type>
class hiplike_kernel_launcher : public rt::backend_kernel_launcher
{
public:
#define __hipsycl_invoke_kernel(nodeptr, f, KernelNameT, KernelBodyT, grid,    \
                                block, shared_mem, stream, ...)                \
  if (false) {                                                                 \
    __hipsycl_kernel_name_template<KernelNameT><<<1, 1>>>();                   \
    __hipsycl_kernel_name_template<KernelBodyT><<<1, 1>>>();                   \
  }                                                                            \
  if constexpr (is_launch_from_module()) {                                     \
    invoke_from_module<KernelNameT, KernelBodyT>(nodeptr, grid, block,         \
                                                 shared_mem, __VA_ARGS__);     \
  } else {                                                                     \
    __hipsycl_launch_integrated_kernel(f, grid, block, shared_mem, stream,     \
                                       __VA_ARGS__)                            \
  }

  hiplike_kernel_launcher()
      : _queue{nullptr}, _invoker{[](rt::dag_node*) {}} {}

  virtual ~hiplike_kernel_launcher() {
    
    for(void* scratch_ptr : _managed_reduction_scratch) {
      // Only assert(_queue) here instead of outside of the loop.
      // If this kernel_launcher was never invoked, it can
      // happen that _queue was never initialized!
      // But if we have reduction scratch allocations, we know
      // that we have been invoked.
      assert(_queue);

      assert(_allocator);
      _allocator->free(scratch_ptr);
    }
  }

  virtual void set_params(void *q) override {
    _queue = reinterpret_cast<Queue_type*>(q);
  }

  Queue_type *get_queue() const {
    return _queue;
  }

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {
    
    this->_type = type;

    using kernel_name_t = typename KernelNameTraits::name;

    sycl::range<Dim> effective_local_range = local_range;
    if constexpr (type == rt::kernel_type::basic_parallel_for) {
      // If local range is non 0, we use it as a hint to override
      // the default selection
      if(local_range.size() == 0) {
        if constexpr (Dim == 1)
          effective_local_range = sycl::range<1>{128};
        else if constexpr (Dim == 2)
          effective_local_range = sycl::range<2>{16, 16};
        else if constexpr (Dim == 3)
          effective_local_range = sycl::range<3>{4, 8, 8};
      }
      HIPSYCL_DEBUG_INFO << "hiplike_kernel_launcher: Submitting high-level "
                            "parallel for with selected total group size of "
                         << effective_local_range.size() << std::endl;
    }

    static constexpr bool has_reductions = sizeof...(Reductions) > 0;

    _invoker = [=](rt::dag_node* node) mutable {
      assert(_queue != nullptr);
      
      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k, reductions...);

      // Simple cases first: Kernel types that don't support
      // reductions
      if constexpr (type == rt::kernel_type::single_task) {

        __hipsycl_invoke_kernel(
            node, hiplike_dispatch::single_task_kernel<kernel_name_t>,
            kernel_name_t, Kernel, dim3(1, 1, 1), dim3(1, 1, 1),
            dynamic_local_memory, _queue->get_native_type(), k);

      } else if constexpr (type == rt::kernel_type::custom) {
       
        sycl::interop_handle handle{_queue->get_device(),
                                    static_cast<void *>(_queue)};

        k(handle);

      } else {
        sycl::range<Dim> grid_range =
            hiplike_dispatch::determine_grid_configuration(
                global_range, effective_local_range);

        std::vector<hiplike_dispatch::reduction_stage<Dim>> reduction_stages;

        if constexpr (has_reductions) {
          reduction_stages = hiplike_dispatch::determine_reduction_stages(
              global_range, effective_local_range, grid_range);
        }

        bool is_with_offset = false;
        for (std::size_t i = 0; i < Dim; ++i)
          if (offset[i] != 0)
            is_with_offset = true;

        auto reducible_kernel_invoker = [&](auto... reduction_descriptors) {

          int required_dynamic_local_mem =
              static_cast<int>(dynamic_local_memory);

          if constexpr (has_reductions) {
            assert(reduction_stages.size() > 0);
            // Proceed to reduction stage 0, i.e. the user-provided kernel
            (reduction_descriptors.proceed_to_stage(
                0, reduction_stages.size(), reduction_stages[0]),
            ...);

            // Reductions will need local memory only *after* the
            // user-provided
            // kernel has completed, so we can reuse the same memory
            required_dynamic_local_mem =
                std::max(required_dynamic_local_mem,
                        reduction_stages[0].allocated_local_memory);
          }

          if constexpr (type == rt::kernel_type::basic_parallel_for) {

            __hipsycl_invoke_kernel(node,
                hiplike_dispatch::parallel_for_kernel<kernel_name_t>, kernel_name_t,
                Kernel,
                hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
                hiplike_dispatch::make_kernel_launch_range<Dim>(
                    effective_local_range),
                required_dynamic_local_mem, _queue->get_native_type(), k,
                global_range, offset, is_with_offset, reduction_descriptors...);

          } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

            for (int i = 0; i < Dim; ++i)
              assert(global_range[i] % effective_local_range[i] == 0);

            __hipsycl_invoke_kernel(node,
                hiplike_dispatch::parallel_for_ndrange_kernel<kernel_name_t>,
                kernel_name_t, Kernel,
                hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
                hiplike_dispatch::make_kernel_launch_range<Dim>(
                    effective_local_range),
                required_dynamic_local_mem, _queue->get_native_type(), k, offset,
                reduction_descriptors...);

          } else if constexpr (type ==
                               rt::kernel_type::hierarchical_parallel_for) {

            for (int i = 0; i < Dim; ++i)
              assert(global_range[i] % effective_local_range[i] == 0);

            __hipsycl_invoke_kernel(node,
                hiplike_dispatch::parallel_for_workgroup<kernel_name_t>,
                kernel_name_t, Kernel,
                hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
                hiplike_dispatch::make_kernel_launch_range<Dim>(
                    effective_local_range),
                required_dynamic_local_mem, _queue->get_native_type(), k,
                effective_local_range, reduction_descriptors...);

          } else if constexpr (type == rt::kernel_type::scoped_parallel_for) {

            for (int i = 0; i < Dim; ++i)
              assert(global_range[i] % effective_local_range[i] == 0);

            auto invoke_scoped_kernel = [&](auto multiversioning_props) {
              using multiversioned_parameters = decltype(multiversioning_props);

              using multiversioned_name_t =
                  typename KernelNameTraits::template multiversioned_name<
                      multiversioned_parameters>;
              
              auto multiversioned_kernel_body =
                  KernelNameTraits::template make_multiversioned_kernel_body<
                      multiversioned_parameters>(k);
              
              using sp_properties_t = decltype(multiversioning_props);

              __hipsycl_invoke_kernel(node,
                  hiplike_dispatch::parallel_region<multiversioned_name_t>,
                  multiversioned_name_t, decltype(multiversioned_kernel_body),
                  hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
                  hiplike_dispatch::make_kernel_launch_range<Dim>(
                      effective_local_range),
                  required_dynamic_local_mem, _queue->get_native_type(),
                  multiversioned_kernel_body, multiversioning_props, grid_range,
                  effective_local_range, reduction_descriptors...);
            };

            if constexpr(Dim == 1) {
              if(effective_local_range[0] % 64 == 0) {
                using sp_properties_t =
                    hiplike_dispatch::sp_multiversioning_properties<64>;
                invoke_scoped_kernel(sp_properties_t{});
              } else if(effective_local_range[0] % 32 == 0) {
                using sp_properties_t =
                    hiplike_dispatch::sp_multiversioning_properties<32>;
                invoke_scoped_kernel(sp_properties_t{});
              } else {
                using sp_properties_t =
                    hiplike_dispatch::sp_multiversioning_properties<1>;
                invoke_scoped_kernel(sp_properties_t{});
              }
            } else {
              using sp_properties_t =
                    hiplike_dispatch::sp_multiversioning_properties<1, 1, 1>;
                invoke_scoped_kernel(sp_properties_t{});
            }

          } else {
            assert(false && "Unsupported kernel type");
          }
          // Launch subsequent dedicated reduction kernels if necessary
          for (std::size_t stage = 1; stage < reduction_stages.size();
               ++stage) {

            (reduction_descriptors.proceed_to_stage(
                 stage, reduction_stages.size(), reduction_stages[stage]),
             ...);
            // Launch dedicated reduction kernel
            //
            // Avoid instantiation of dedicated reduction kernel
            // if there are no reductions for compatibility reasons
            // (requires unique lambda name mangling if the reduction
            //  combiner is a lambda function)
            if constexpr (has_reductions) {
              std::size_t num_groups =
                  reduction_stages[stage].num_groups.size();
              std::size_t local_size =
                  reduction_stages[stage].local_size.size();

              int num_elements = reduction_stages[stage].global_size.size();

              auto pure_reduction_kernel = [=](int gid,
                                               auto &...local_reducers) {
                if (gid < num_elements)
                  (local_reducers.combine_global_input(gid), ...);
              };

              __hipsycl_invoke_kernel(node,
                  hiplike_dispatch::primitive_parallel_for_with_local_reducers<
                      __hipsycl_unnamed_kernel>,
                  __hipsycl_unnamed_kernel, decltype(pure_reduction_kernel),
                  hiplike_dispatch::make_kernel_launch_range<1>(
                      sycl::range<1>{num_groups}),
                  hiplike_dispatch::make_kernel_launch_range<1>(
                      sycl::range<1>{local_size}),
                  reduction_stages[stage].allocated_local_memory,
                  _queue->get_native_type(), pure_reduction_kernel,
                  reduction_descriptors...);
            }
          }
        };

        _allocator = node->get_runtime()
                         ->backends()
                         .get(_queue->get_device().get_backend())
                         ->get_allocator(_queue->get_device());

        hiplike_dispatch::invoke_reducible_kernel(
            reducible_kernel_invoker, reduction_stages,
            _managed_reduction_scratch, _allocator, reductions...);
      }
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return Backend_id;
  }

  virtual void invoke(rt::dag_node* node) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:
  
  static constexpr bool is_launch_from_module() {

    constexpr auto is_cuda_module_launch = [](){
#ifdef __HIPSYCL_MULTIPASS_CUDA_HEADER__
      return Backend_id == rt::backend_id::cuda;
#else
      return false;
#endif
    };

    constexpr auto is_hip_module_launch = [](){
#ifdef __HIPSYCL_MULTIPASS_HIP_HEADER__
      return Backend_id == rt::backend_id::hip;
#else
      return false;
#endif
    };

    return is_cuda_module_launch() || is_hip_module_launch();
  }

  template<class KernelT>
  std::string get_stable_kernel_name() const {
    // On clang 11, rely on __builtin_unique_stable_name()
#if defined(__clang__) && __clang_major__ == 11

    return __builtin_unique_stable_name(KernelT);

    // If we have builtin_get_device_side_mangled_name, rely on
    // __hipsycl_kernel_name_template unless we are in split compiler mode -
    // here we follow the traditional mangling based on the type.
    // In thas case, unnamed kernel lambdas are unsupported which is enforced
    // by the clang plugin in the device compilation pass.
#elif __has_builtin(__builtin_get_device_side_mangled_name) &&                 \
    !defined(__HIPSYCL_SPLIT_COMPILER__)
    
    // The builtin unfortunately only works with __global__ or
    // __device__ functions. Since our kernel launchers cannot be __global__
    // when semantic analysis runs, we cannot apply the builtin
    // directly to our kernel launchers. Use dummy __global__ instead
    std::string name_template = __builtin_get_device_side_mangled_name(
      __hipsycl_kernel_name_template<KernelT>);
    std::string template_marker = "_Z30__hipsycl_kernel_name_template";
    std::string replacement = "_Z16__hipsycl_kernel";
    name_template.erase(0, template_marker.size());
    return replacement + name_template;
#else
    return typeid(KernelT).name();
#endif
  }

  template <class KernelName, class KernelBodyT, typename... Args>
  void invoke_from_module(rt::dag_node* node, dim3 grid_size, dim3 block_size,
                          unsigned dynamic_shared_mem, Args... args) {
    assert(node);
  
#if defined(__HIPSYCL_MULTIPASS_CUDA_HEADER__) || defined(__HIPSYCL_MULTIPASS_HIP_HEADER__)

    std::size_t local_hcf_object_id = 0;
#ifdef __HIPSYCL_MULTIPASS_CUDA_HEADER__
    if(Backend_id == rt::backend_id::cuda) {
      local_hcf_object_id = __hipsycl_local_cuda_hcf_object_id;
    }
#endif
#ifdef __HIPSYCL_MULTIPASS_HIP_HEADER__
    if(Backend_id == rt::backend_id::hip) {
      local_hcf_object_id = __hipsycl_local_hip_hcf_object_id;
    }
#endif

    std::array<void *, sizeof...(Args)> kernel_args{
      static_cast<void *>(&args)...
    };
    std::array<std::size_t, sizeof...(Args)> arg_sizes{
      sizeof(Args)...
    };

    std::string kernel_name_tag = get_stable_kernel_name<KernelName>();
    std::string kernel_body_name = get_stable_kernel_name<KernelBodyT>();

    rt::code_object_invoker *invoker = _queue->get_code_object_invoker();

    assert(invoker &&
            "Runtime backend does not support invoking kernels from modules");

    auto num_groups = rt::range<3>{grid_size.x, grid_size.y, grid_size.z};
    auto group_size = rt::range<3>{block_size.x, block_size.y, block_size.z};

    const rt::kernel_operation &op =
        *static_cast<rt::kernel_operation*>(node->get_operation());

    rt::result err = invoker->submit_kernel(op, local_hcf_object_id,
        num_groups, group_size, dynamic_shared_mem, kernel_args.data(),
        arg_sizes.data(), kernel_args.size(), kernel_name_tag,
        kernel_body_name);

    if (!err.is_success())
      rt::register_error(err);
#else
    assert(false && "No module available to invoke kernels from");
#endif
  }

  Queue_type *_queue;
  rt::kernel_type _type;
  std::function<void (rt::dag_node*)> _invoker;

  std::vector<void*> _managed_reduction_scratch;
  rt::backend_allocator* _allocator = nullptr;
};

}
}

#undef __hipsycl_invoke_kernel
#undef __sycl_kernel

#endif
