/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2021 Aksel Alpay and contributors
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

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "exception.hpp"
#include "access.hpp"
#include "context.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "libkernel/backend.hpp"
#include "device.hpp"
#include "event.hpp"
#include "types.hpp"
#include "usm_query.hpp"
#include "libkernel/accessor.hpp"
#include "libkernel/builtin_kernels.hpp"
#include "libkernel/id.hpp"
#include "libkernel/range.hpp"
#include "libkernel/nd_range.hpp"
#include "libkernel/item.hpp"
#include "libkernel/nd_item.hpp"
#include "libkernel/group.hpp"
#include "libkernel/detail/local_memory_allocator.hpp"
#include "detail/util.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/glue/embedded_pointer.hpp"
#include "hipSYCL/glue/kernel_launcher_factory.hpp"
#include "hipSYCL/glue/kernel_names.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template <int Dim> struct accessor_data {
  std::shared_ptr<rt::buffer_data_region> mem;
  
  sycl::id<Dim> offset;
  sycl::range<Dim> range;

  bool is_no_init;
};


} // namespace detail

class queue;

class handler {
  friend class queue;

  template <class AccessorType, int Dim>
  friend void
  detail::accessor::bind_to_handler(AccessorType &acc, sycl::handler &cgh,
                                    std::shared_ptr<rt::buffer_data_region> mem,
                                    sycl::id<Dim> offset, sycl::range<Dim> range,
                                    bool is_no_init);

  template <class AccessorType, int Dim>
  void require(AccessorType& acc,
               const detail::accessor_data<Dim>& data) {

    glue::unique_id accessor_id = acc.get_uid();

    if (!data.mem) {
      throw invalid_parameter_error{
          "handler: require(): accessor is illegal paramater for require() "
          "because it is not bound to a buffer."};
    }

    // Translate no_init property and host_task modes
    access_mode mode =
        detail::get_effective_access_mode(AccessorType::mode, data.is_no_init);
    
    size_t element_size = data.mem->get_element_size();

    const rt::range<Dim> buffer_shape = rt::make_range(acc.get_buffer_shape());
    
    auto req = std::make_unique<rt::buffer_memory_requirement>(
      data.mem,
      detail::get_effective_offset<typename AccessorType::value_type>(
          data.mem, rt::make_id(data.offset), buffer_shape,
          AccessorType::has_access_range),
      detail::get_effective_range<typename AccessorType::value_type>(
          data.mem, rt::make_range(data.range), buffer_shape,
          AccessorType::has_access_range),
      mode, AccessorType::access_target
    );

    // Bind the accessor's embedded pointer to the requirement, such that
    // the scheduler is able to initialize the accessor's data pointer
    // once it has been captured
    req->bind(accessor_id);

    _requirements.add_requirement(std::move(req));
  }

  template <typename dataT, int dimensions, access_mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  static constexpr std::size_t get_dimensions(
      accessor<dataT, dimensions, accessMode, accessTarget, isPlaceholder>
          acc) {
    return dimensions;
  }

  template<class AccessorType>
  void raise_unregistered_accessor(const AccessorType&) {

    HIPSYCL_DEBUG_ERROR << "Attempted to access accessor that was not "
                           "registered with the handler"
                        << std::endl;
    throw sycl::invalid_parameter_error{
        "Accessor was not registered with handler"};
  }

  template <class AccessorType>
  rt::buffer_memory_requirement *
  get_buffer_memory_requirement(const AccessorType &acc) {
    rt::buffer_memory_requirement *req = nullptr;

    for (rt::dag_node_ptr req : _requirements.get()) {

      if (req->get_operation()->is_requirement()) {
        if (rt::cast<rt::requirement>(req->get_operation())
                ->is_memory_requirement()) {
          rt::memory_requirement *mreq =
              rt::cast<rt::memory_requirement>(req->get_operation());
          if (mreq->is_buffer_requirement()) {
            rt::buffer_memory_requirement *bmem_req =
                rt::cast<rt::buffer_memory_requirement>(req->get_operation());
            if (bmem_req->is_bound()) {
              if (bmem_req->get_bound_accessor_id() == acc.get_uid())
                return bmem_req;
            }
          }
        }
      }
    }

    return nullptr;
  }

  template<int Dim>
  sycl::id<Dim> rt_id_to_sycl_id(rt::id<Dim> in){
    sycl::id<Dim> out;
    for(int i = 0; i < Dim; ++i) {
      out[i] = in[i];
    }
    return out;
  }

  template<int Dim>
  sycl::range<Dim> rt_range_to_sycl_range(rt::range<Dim> in){
    sycl::range<Dim> out;
    for(int i = 0; i < Dim; ++i) {
      out[i] = in[i];
    }
    return out;
  }

  template<class AccessorType>
  auto get_offset(const AccessorType& acc) {
    rt::buffer_memory_requirement *mem_req = get_buffer_memory_requirement(acc);

    if(!mem_req)
      raise_unregistered_accessor(acc);

    return rt_id_to_sycl_id(
        rt::extract_from_id3<AccessorType::get_dimensions()>(
            mem_req->get_access_offset3d()));
  }

  template <class AccessorType> auto get_range(const AccessorType &acc) {
     rt::buffer_memory_requirement *mem_req = get_buffer_memory_requirement(acc);

    if(!mem_req)
      raise_unregistered_accessor(acc);

    return rt_range_to_sycl_range(
        rt::extract_from_range3<AccessorType::get_dimensions()>(
            mem_req->get_access_range3d()));
  }

  template<class AccessorType>
  auto get_memory_region(const AccessorType& acc) {
    rt::buffer_memory_requirement *mem_req = get_buffer_memory_requirement(acc);

    if(!mem_req)
      raise_unregistered_accessor(acc);
    
    return mem_req->get_data_region();
  }

public:
  ~handler()
  {
  }

  template <typename dataT, int dimensions, access_mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  void
  require(accessor<dataT, dimensions, accessMode, accessTarget, isPlaceholder>
              acc) {
    static_assert(accessTarget != access::target::local,
                  "Requiring local accessors is unsupported");

    // TODO: Prevent require() on slimmed down accessor

    // Construct requirement descriptor
    std::shared_ptr<rt::buffer_data_region> data_region =
        acc.get_data_region();

    if(!data_region) {
      throw invalid_parameter_error{
          "handler: require(): accessor is illegal paramater for require() "
          "because it is not bound to a buffer."};
    }

    auto offset = acc.get_offset();
    auto range = acc.get_range();

    using AccessorT =
        accessor<dataT, dimensions, accessMode, accessTarget, isPlaceholder>;

    detail::accessor_data<dimensions> data{
      acc.get_data_region(),
      offset,
      range,
      acc.is_no_init()
    };

    this->require(acc, data);
  }

  void depends_on(event e) {
    // No need to consider default constructed events that are
    // not bound to a node
    if(e._node) {
      _requirements.add_node_requirement(e._node);
    }
  }

  void depends_on(const std::vector<event> &events) {
    for (auto e : events) {
      depends_on(e);
    }
  }


  template <typename KernelName = __hipsycl_unnamed_kernel, typename KernelType>
  void single_task(KernelType kernelFunc)
  {
    this->submit_kernel<KernelName, rt::kernel_type::single_task>(
      sycl::id<1>{0}, sycl::range<1>{1}, sycl::range<1>{1}, kernelFunc);
  }

  template <typename KernelName = __hipsycl_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    const ReductionsAndKernel &... redu_kernel) {

    auto invoker = [&](auto&& kernel, auto&&... reductions){
      this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
          sycl::id<dimensions>{}, numWorkItems,
          get_preferred_group_size<dimensions>(),
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  template <typename KernelName = __hipsycl_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset,
                    const ReductionsAndKernel &... redu_kernel) {
    auto invoker = [&](auto&& kernel, auto&& ... reductions) {
      this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
          workItemOffset, numWorkItems,
          get_preferred_group_size<dimensions>(),
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  template <typename KernelName = __hipsycl_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange,
                    const ReductionsAndKernel &... redu_kernel) {
    auto invoker = [&](auto&& kernel, auto&& ... reductions) {
      this->submit_kernel<KernelName, rt::kernel_type::ndrange_parallel_for>(
          executionRange.get_offset(), executionRange.get_global_range(),
          executionRange.get_local_range(),
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  // Hierarchical kernel dispatch API

  /// \todo flexible ranges are currently unsupported
  /*
  template <typename KernelName= __hipsycl_unnamed_kernel,
            typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               WorkgroupFunctionType kernelFunc)
  {
    dispatch_hierarchical_kernel(numWorkGroups,
                                 get_default_local_range<dimensions>(),
                                 kernelFunc);
  }
  */

  template <typename KernelName = __hipsycl_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               range<dimensions> workGroupSize,
                               const ReductionsAndKernel &... redu_kernel) {
    auto invoker = [&](auto &&kernel, auto &&... reductions) {
      this->submit_kernel<KernelName,
                          rt::kernel_type::hierarchical_parallel_for>(
          sycl::id<dimensions>{}, numWorkGroups * workGroupSize, workGroupSize,
          kernel, reductions...);
    };
    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  // Scoped parallelism API
  
  template <typename KernelName = __hipsycl_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  void parallel(range<dimensions> numWorkGroups,
                range<dimensions> workGroupSize,
                const ReductionsAndKernel& ... redu_kernel)
  {
    auto invoker = [&](auto&& kernel, auto&&... reductions) {
      this->submit_kernel<KernelName, rt::kernel_type::scoped_parallel_for>(
          sycl::id<dimensions>{}, numWorkGroups * workGroupSize,
          workGroupSize,
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
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

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void copy(accessor<T, dim, mode, tgt, variant> src, shared_ptr_class<T> dest) {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void copy(shared_ptr_class<T> src, accessor<T, dim, mode, tgt, variant> dest) {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void copy(accessor<T, dim, mode, tgt, variant> src, T *dest) {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void copy(const T *src, accessor<T, dim, mode, tgt, variant> dest) {
    copy_ptr(src, dest);
  }

  template <typename T, int dim, access::mode srcMode, access::mode dstMode,
            access::target srcTgt, access::target destTgt,
            accessor_variant VariantSrc, accessor_variant VariantDest>
  void copy(accessor<T, dim, srcMode, srcTgt, VariantSrc> src,
            accessor<T, dim, dstMode, destTgt, VariantDest> dest)
  {
    validate_copy_src_accessor(src);
    validate_copy_dest_accessor(dest);

    for(int i = 0; i < dim; ++i)
    {
      if(get_range(src).get(i) > get_range(dest).get(i))
      {
        throw invalid_parameter_error{"handler: copy(): "
          "Accessor sizes are incompatible."};
      }
    }

    std::shared_ptr<rt::buffer_data_region> data_src  = get_memory_region(src);
    std::shared_ptr<rt::buffer_data_region> data_dest = get_memory_region(dest);

    if (sizeof(T) != data_src->get_element_size() ||
        sizeof(T) != data_dest->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit copy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id src_dev = get_explicit_accessor_target(src);
    rt::device_id dest_dev = get_explicit_accessor_target(dest);
    
    rt::memory_location source_location{src_dev, rt::embed_in_id3(get_offset(src)),
                                        data_src};
    rt::memory_location dest_location{dest_dev, rt::embed_in_id3(get_offset(dest)),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(src)));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void update_host(accessor<T, dim, mode, tgt, variant> acc) {
    update_dev(detail::get_host_device(), acc);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void update(accessor<T, dim, mode, tgt, variant> acc) {
    
    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: device update() is unsupported "
                                    "for queues not bound to devices"};

    update_dev(
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id(),
        acc);
  }

  /// \todo fill() on host accessors can be optimized to use
  /// memset() if the accessor describes a large area of
  /// contiguous memory
  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void fill(accessor<T, dim, mode, tgt, variant> dest, const T &src) {
    static_assert(mode != access::mode::read,
                  "Filling read-only accessors is not allowed.");
    static_assert(tgt != access::target::host_image,
                  "host_image targets are unsupported");


    this->submit_kernel<__hipsycl_unnamed_kernel, rt::kernel_type::basic_parallel_for>(
        get_offset(dest), get_range(dest),
        get_preferred_group_size<dim>(),
        detail::kernels::fill_kernel{dest, src});
  }

  // ------ USM functions ------

  void memcpy(void *dest, const void *src, std::size_t num_bytes) {

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit memcpy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id queue_dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();
  

    auto determine_ptr_device = [&, this](const void *ptr) {
      usm::alloc alloc_type = get_pointer_type(ptr, _ctx);
      // For shared allocations, be optimistic and assume that data is
      // already on target device
      if (alloc_type == usm::alloc::shared)
        return queue_dev;

      if (alloc_type == usm::alloc::host ||
          alloc_type == usm::alloc::unknown)
        return detail::get_host_device();

      if(alloc_type == usm::alloc::device)
        // we are dealing with a device allocation
        return detail::extract_rt_device(get_pointer_device(ptr, _ctx));
      
      throw invalid_parameter_error{"Invalid allocation type"};
    };

    rt::device_id src_dev = determine_ptr_device(src);
    rt::device_id dest_dev = determine_ptr_device(dest);
    
    rt::memory_location source_location{
        src_dev, extract_ptr(src), rt::id<3>{},
        rt::embed_in_range3(range<1>{num_bytes}), 1};

    rt::memory_location dest_location{
        dest_dev, extract_ptr(dest), rt::id<3>{},
        rt::embed_in_range3(range<1>{num_bytes}), 1};
    
    auto op = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(range<1>{num_bytes}));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(op), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }
  
  template <typename T>
  void copy(const T* src, T* dest, std::size_t count) {
    this->memcpy(static_cast<void*>(dest),
      static_cast<const void*>(src), count * sizeof(T));
  }


  template <class T> void fill(void *ptr, const T &pattern, std::size_t count) {
    // For special cases we can map this to a potentially more low-level memset
    if (sizeof(T) == 1) {
      unsigned char val = *reinterpret_cast<const unsigned char*>(&pattern);
      
      memset(ptr, static_cast<int>(val), count);
    } else {
      T *typed_ptr = static_cast<T *>(ptr);

      if (!_execution_hints.has_hint<rt::hints::bind_to_device>())
        throw invalid_parameter_error{"handler: USM fill() is unsupported "
                                      "for queues not bound to devices"};

      this->submit_kernel<__hipsycl_unnamed_kernel,
                          rt::kernel_type::basic_parallel_for>(
          sycl::id<1>{}, sycl::range<1>{count},
          get_preferred_group_size<1>(),
          detail::kernels::fill_kernel_usm{typed_ptr, pattern});
    }
  }

  void memset(void *ptr, int value, std::size_t num_bytes) {
   
    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit memset() is unsupported "
                                    "for queues not bound to devices"};

    auto op = rt::make_operation<rt::memset_operation>(
        ptr, static_cast<unsigned char>(value), num_bytes);

    rt::dag_node_ptr node = build.builder()->add_memset(
        std::move(op), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  void prefetch_host(const void *ptr, std::size_t num_bytes) {

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit prefetch() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id executing_dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();

    // The device to which we prefetch
    rt::device_id target_dev = detail::get_host_device();

    // If this queue is on the host, we need to execute on the backend
    // that has provided the USM pointer instead.
    rt::execution_hints hints = _execution_hints;

    if (executing_dev.is_host()) {

      auto usm_dev = detail::extract_rt_device(get_pointer_device(ptr, _ctx));

      hints.overwrite_with(rt::make_execution_hint<rt::hints::bind_to_device>(
          usm_dev));
    }

    auto op = rt::make_operation<rt::prefetch_operation>(
        ptr, num_bytes, target_dev);

    rt::dag_node_ptr node = build.builder()->add_prefetch(
        std::move(op), _requirements, hints);

    _command_group_nodes.push_back(node);
  }

  void prefetch(const void *ptr, std::size_t num_bytes) {

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit prefetch() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id executing_dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();

    if (executing_dev.is_host())
      // If this queue is bound to a host device, run prefetch_host()
      // to potentially get data back to host.
      prefetch_host(ptr, num_bytes);
    else {
      // Otherwise, run prefetch on the queue's device to the
      // queue's device
      rt::dag_build_guard build{rt::application::dag()};

      auto op = rt::make_operation<rt::prefetch_operation>(
          ptr, num_bytes, executing_dev);

      rt::dag_node_ptr node = build.builder()->add_prefetch(
          std::move(op), _requirements, _execution_hints);

      _command_group_nodes.push_back(node);
    }
  }

  void mem_advise(const void *addr, std::size_t num_bytes, int advice) {
    throw feature_not_supported{"mem_advise() is not yet supported"};
  }


  template <class InteropFunction>
  void hipSYCL_enqueue_custom_operation(InteropFunction f) {
    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{
          "handler: submitting custom operations is unsupported "
          "for queues not bound to devices"};

    rt::dag_build_guard build{rt::application::dag()};

    auto custom_kernel_op = rt::make_operation<rt::kernel_operation>(
        typeid(f).name(),
        glue::make_kernel_launchers<class _unnamed, rt::kernel_type::custom>(
            sycl::id<3>{}, sycl::range<3>{}, 
            sycl::range<3>{},
            0, f),
        _requirements);

    rt::dag_node_ptr node = build.builder()->add_kernel(
        std::move(custom_kernel_op), _requirements, _execution_hints);
    
    _command_group_nodes.push_back(node);
  }
  
  detail::local_memory_allocator& get_local_memory_allocator()
  {
    return _local_mem_allocator;
  }

private:
  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  rt::device_id get_explicit_accessor_target(
      const accessor<T, dim, mode, tgt, variant> &acc) {
    if (tgt == access::target::host_buffer)
      return detail::get_host_device();
    assert(_execution_hints.has_hint<rt::hints::bind_to_device>());
    return _execution_hints.get_hint<rt::hints::bind_to_device>()
        ->get_device_id();
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void update_dev(rt::device_id dev, accessor<T, dim, mode, tgt, variant> acc) {
    HIPSYCL_DEBUG_INFO
        << "handler: Spawning async generalized device update task"
        << std::endl;

    std::shared_ptr<rt::buffer_data_region> data = get_memory_region(acc);

    if(!data)
      throw sycl::invalid_parameter_error{
          "update_dev(): Accessor is not bound to buffer"};

    rt::dag_build_guard build{rt::application::dag()};

    const rt::range<dim> buffer_shape = rt::make_range(acc.get_buffer_shape());
    constexpr bool has_access_range =
      accessor<T, dim, mode, tgt, variant>::has_access_range;

    auto explicit_requirement = rt::make_operation<rt::buffer_memory_requirement>(
      data,
      detail::get_effective_offset<T>(data, rt::make_id(get_offset(acc)),
                                      buffer_shape, has_access_range),
      detail::get_effective_range<T>(data, rt::make_range(get_range(acc)),
                                      buffer_shape, has_access_range),
      mode, tgt
    );

    rt::execution_hints enforce_bind_to_dev;
    enforce_bind_to_dev.add_hint(
        rt::make_execution_hint<rt::hints::bind_to_device>(
            dev));

    // Merge new hint into default hints
    rt::execution_hints hints = _execution_hints;
    hints.overwrite_with(enforce_bind_to_dev);
    assert(hints.has_hint<rt::hints::bind_to_device>());
    assert(hints.get_hint<rt::hints::bind_to_device>()->get_device_id() ==
           dev);

    rt::dag_node_ptr node = build.builder()->add_explicit_mem_requirement(
        std::move(explicit_requirement), _requirements, hints);

    _command_group_nodes.push_back(node);
  }

  template <class KernelName, rt::kernel_type KernelType, class KernelFuncType,
            int Dim, typename... Reductions>
  void submit_kernel(sycl::id<Dim> offset, sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range, KernelFuncType f,
                     Reductions... reductions) {
    std::size_t shared_mem_size = _local_mem_allocator.get_allocation_size();

    rt::dag_build_guard build{rt::application::dag()};

    auto kernel_op = rt::make_operation<rt::kernel_operation>(
        typeid(f).name(),
        glue::make_kernel_launchers<KernelName, KernelType>(
            offset, local_range, global_range, shared_mem_size, f,
            reductions...),
        _requirements);

    rt::dag_node_ptr node = build.builder()->add_kernel(
        std::move(kernel_op), _requirements, _execution_hints);
    
    _command_group_nodes.push_back(node);
  }

  template<class T>
  void* extract_ptr(std::shared_ptr<T> ptr)
  { return reinterpret_cast<void*>(ptr.get()); }

  template<class T>
  void* extract_ptr(T* ptr)
  { return reinterpret_cast<void*>(ptr); }

  template<class T>
  void* extract_ptr(const T* ptr)
  { return extract_ptr(const_cast<T*>(ptr)); }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant, typename destPtr>
  void copy_ptr(accessor<T, dim, mode, tgt, variant> src, destPtr dest) {
    validate_copy_src_accessor(src);

    std::shared_ptr<rt::buffer_data_region> data_src = get_memory_region(src);

    if (sizeof(T) != data_src->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit copy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(src);

    rt::memory_location source_location{dev, rt::embed_in_id3(get_offset(src)),
                                        data_src};
    // Assume the allocation behind dest is large enough to hold src.get_range.size() contiguous elements
    rt::memory_location dest_location{detail::get_host_device(), extract_ptr(dest),
                                      rt::id<3>{}, rt::embed_in_range3(get_range(src)),
                                      data_src->get_element_size()};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(src)));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant, typename srcPtr>
  void copy_ptr(srcPtr src, accessor<T, dim, mode, tgt, variant> dest) {
    validate_copy_dest_accessor(dest);

    std::shared_ptr<rt::buffer_data_region> data_dest = get_memory_region(dest);

    if (sizeof(T) != data_dest->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit copy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(dest);

    // Assume src contains src.get_range.size() contiguous elements
    rt::memory_location source_location{detail::get_host_device(), extract_ptr(src),
                                      rt::id<3>{}, rt::embed_in_range3(get_range(dest)),
                                      data_dest->get_element_size()};
    rt::memory_location dest_location{dev, rt::embed_in_id3(get_offset(dest)),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(dest)));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void validate_copy_src_accessor(const accessor<T, dim, mode, tgt, variant> &) {
    static_assert(dim != 0, "0-dimensional accessors are currently not supported");
    static_assert(mode == access::mode::read || mode == access::mode::read_write,
      "Only read or read_write accessors can be copied from");
    static_assert(tgt == access::target::global_buffer ||
      tgt == access::target::host_buffer,
      "Only global_buffer or host_buffer accessors are currently "
      "supported for copying");
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            accessor_variant variant>
  void validate_copy_dest_accessor(const accessor<T, dim, mode, tgt, variant> &) {
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

  const std::vector<rt::dag_node_ptr>& get_cg_nodes() const
  { return _command_group_nodes; }

  
  handler(const context &ctx, async_handler handler,
          const rt::execution_hints &hints)
      : _ctx{ctx}, _handler{handler}, _execution_hints{hints},
        _preferred_group_size1d{}, _preferred_group_size2d{},
        _preferred_group_size3d{} {}

  template<int Dim>
  range<Dim>& get_preferred_group_size() {
    if constexpr(Dim == 1) {
      return _preferred_group_size1d;
    } else if constexpr(Dim == 2) {
      return _preferred_group_size2d;
    } else {
      return _preferred_group_size3d;
    }
  }

  template<int Dim>
  const range<Dim>& get_preferred_group_size() const {
    if constexpr(Dim == 1) {
      return _preferred_group_size1d;
    } else if constexpr(Dim == 2) {
      return _preferred_group_size2d;
    } else {
      return _preferred_group_size3d;
    }
  }
  
  template<int Dim>
  void set_preferred_group_size(range<Dim> r) {
    get_preferred_group_size<Dim>() = r;
  }
  
  const context _ctx;
  detail::local_memory_allocator _local_mem_allocator;
  async_handler _handler;

  rt::requirements_list _requirements;
  rt::execution_hints _execution_hints;
  std::vector<rt::dag_node_ptr> _command_group_nodes;

  range<1> _preferred_group_size1d;
  range<2> _preferred_group_size2d;
  range<3> _preferred_group_size3d;
};

namespace detail::handler {

template<class T>
inline local_memory::address allocate_local_mem(
    sycl::handler& cgh,
    size_t num_elements)
{
  return cgh.get_local_memory_allocator().alloc<T>(num_elements);
}

}

namespace detail::accessor {

template<class AccessorType>
void bind_to_handler(AccessorType& acc, sycl::handler& cgh)
{
  cgh.require(acc);
}

template <class AccessorType, int Dim>
void bind_to_handler(AccessorType& acc, sycl::handler& cgh,
                     std::shared_ptr<rt::buffer_data_region> mem,
                     sycl::id<Dim> offset, sycl::range<Dim> range, bool is_no_init) {
  cgh.require(acc, detail::accessor_data<Dim>{mem, offset, range, is_no_init});
}

}

} // namespace sycl
} // namespace hipsycl

#endif
