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
#include "hipSYCL/sycl/device.hpp"
#include "types.hpp"
#include "id.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "item.hpp"
#include "nd_item.hpp"
#include "group.hpp"
#include "detail/util.hpp"
#include "detail/local_memory_allocator.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/glue/kernel_launcher_factory.hpp"

namespace hipsycl {
namespace sycl {


class queue;

class handler
{
  friend class sycl::queue;
public:
  ~handler()
  {
  }

  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  void
  require(accessor<dataT, dimensions, accessMode, accessTarget, isPlaceholder>&
              acc) {
    static_assert(accessTarget != access::target::local,
                  "Requiring local accessors is unsupported");
    
    // Construct requirement descriptor
    std::shared_ptr<rt::buffer_data_region> data_region = acc._buff.get_shared_ptr();
    
    auto offset = acc.get_offset();
    auto range = acc.get_range();
    size_t element_size = data_region->get_element_size();

    if(element_size != sizeof(dataT)) {
      assert(false && "Reinterpreting data with elements of different size is "
                      "not yet supported");
    }

    auto req = std::make_unique<rt::buffer_memory_requirement>(
        data_region, rt::make_id(offset), rt::make_range(range), accessMode,
        accessTarget);

    // Bind the accessor's deferred pointer to the requirement, such that
    // the scheduler is able to initialize the accessor's data pointer
    // once it has been captured
    acc.bind_to(req.get());

    _requirements.add_requirement(std::move(req));
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
    this->submit_kernel<KernelName, rt::kernel_type::single_task>(
      sycl::id<1>{0}, sycl::range<1>{1}, sycl::range<1>{1}, kernelFunc);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc)
  {
    this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
        sycl::id<dimensions>{}, numWorkItems,
        numWorkItems /* local range is ignored for basic parallel for*/,
        kernelFunc);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, KernelType kernelFunc)
  {
    this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
        workItemOffset, numWorkItems,
        numWorkItems /* local range is ignored for basic parallel for*/,
        kernelFunc);
  }

  template <typename KernelName = class _unnamed_kernel,
            typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernelFunc)
  {
    this->submit_kernel<KernelName, rt::kernel_type::ndrange_parallel_for>(
        executionRange.get_offset(), executionRange.get_global_range(),
        executionRange.get_local_range(),
        kernelFunc);
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
    this->submit_kernel<KernelName, rt::kernel_type::hierarchical_parallel_for>(
        sycl::id<dimensions>{}, numWorkGroups * workGroupSize,
        workGroupSize,
        kernelFunc);
  }

  // Scoped parallelism API
  
  template <typename KernelName = class _unnamed_kernel,
            typename KernelFunctionType, int dimensions>
  void parallel(range<dimensions> numWorkGroups,
                range<dimensions> workGroupSize,
                KernelFunctionType f)
  {
    this->submit_kernel<KernelName, rt::kernel_type::scoped_parallel_for>(
        sycl::id<dimensions>{}, numWorkGroups * workGroupSize,
        workGroupSize,
        f);
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
    validate_copy_src_accessor(src);
    validate_copy_dest_accessor(dest);

    for(int i = 0; i < dim; ++i)
    {
      if(src.get_range().get(i) > dest.get_range().get(i))
      {
        throw invalid_parameter_error{"handler: copy(): "
          "Accessor sizes are incompatible."};
      }
    }

    std::shared_ptr<rt::buffer_data_region> data_src  = src._buff.get_shared_ptr();
    std::shared_ptr<rt::buffer_data_region> data_dest = dest._buff.get_shared_ptr();

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
    
    rt::memory_location source_location{src_dev, rt::embed_in_id3(src.get_offset()),
                                        data_src};
    rt::memory_location dest_location{dest_dev, rt::embed_in_id3(dest.get_offset()),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(src.get_range()));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt>
  void update_host(accessor<T, dim, mode, tgt> acc)
  {
    HIPSYCL_DEBUG_INFO << "handler: Spawning async host access task"
                       << std::endl;

    if(!acc._buff)
      throw sycl::invalid_parameter_error{
          "update_host(): Accessor is not bound to buffer"};

    std::shared_ptr<rt::buffer_data_region> data = acc._buff.get_shared_ptr();

    if(sizeof(T) != data->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    auto explicit_requirement = rt::make_operation<rt::buffer_memory_requirement>(
        data, acc.get_offset(), acc.get_range(), mode, tgt);

    rt::execution_hints enforce_bind_to_host;
    enforce_bind_to_host.add_hint(
        rt::make_execution_hint<rt::hints::bind_to_device>(
            detail::get_host_device()));

    // Merge new hint into default hints
    rt::execution_hints hints = _execution_hints;
    hints.overwrite_with(enforce_bind_to_host);
    assert(hints.has_hint<rt::hints::bind_to_device>());
    assert(hints.get_hint<rt::hints::bind_to_device>()->get_device_id() ==
           detail::get_host_device());

    rt::dag_node_ptr node = build.builder()->add_explicit_mem_requirement(
        std::move(explicit_requirement), _requirements, hints);

    _command_group_nodes.push_back(node);
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

    // Use a function object instead of lambda to avoid
    // requiring a unique kernel name for each fill call

    // TODO: hipSYCL rt currently does not have a dedicated operation
    // for fills - implement for the ability to implement fill using
    // backend functionality instead of a kernel
    class fill_kernel
    {
    public:
      fill_kernel(accessor<T, dim, mode, tgt> dest,
                  const T& src)
      : _dest{dest}, _src{src}
      {}

      void operator()(sycl::id<dim> tid)
      {
        _dest[tid] = _src;
      }

    private:
      accessor<T, dim, mode, tgt> _dest;
      T _src;
    };

    this->submit_kernel<class _unnamed_kernel, rt::kernel_type::basic_parallel_for>(
        dest.get_offset(), dest.get_range(),
        dest.get_range() /*local range unused for basic pf*/,
        fill_kernel{dest, src});
  }

  detail::local_memory_allocator& get_local_memory_allocator()
  {
    return _local_mem_allocator;
  }

private:
  template <typename T, int dim, access::mode mode, access::target tgt>
  rt::device_id
  get_explicit_accessor_target(const accessor<T, dim, mode, tgt> &acc) {
    if (tgt == access::target::host_buffer)
      return detail::get_host_device();
    assert(_execution_hints.has_hint<rt::hints::bind_to_device>());
    return _execution_hints.get_hint<rt::hints::bind_to_device>()
        ->get_device_id();
  }
  template <
    class KernelName, rt::kernel_type KernelType, class KernelFuncType, int Dim>
  void submit_kernel(sycl::id<Dim> offset, sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range, KernelFuncType f) {
    std::size_t shared_mem_size = _local_mem_allocator.get_allocation_size();

    rt::dag_build_guard build{rt::application::dag()};

    auto kernel_op = rt::make_operation<rt::kernel_operation>(
        typeid(f).name(),
        glue::make_kernel_launchers<KernelName, KernelType>(
            offset, local_range, 
            global_range,
            shared_mem_size, f),
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
            typename destPtr>
  void copy_ptr(accessor<T, dim, mode, tgt> src, destPtr dest)
  {
    validate_copy_src_accessor(src);

    std::shared_ptr<rt::buffer_data_region> data_src = src._buff.get_shared_ptr();

    if (sizeof(T) != data_src->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit copy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(src);

    rt::memory_location source_location{dev, rt::embed_in_id3(src.get_offset()),
                                        data_src};
    // Assume dest element size and allocation shape is the same as src
    rt::memory_location dest_location{detail::get_host_device(), extract_ptr(dest),
                                      rt::id<3>{}, data_src->get_num_elements(),
                                      data_src->get_element_size()};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(src.get_range()));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            typename srcPtr>
  void copy_ptr(srcPtr src, accessor<T, dim, mode, tgt> dest)
  {
    validate_copy_dest_accessor(dest);

    std::shared_ptr<rt::buffer_data_region> data_dest = dest._buff.get_shared_ptr();

    if (sizeof(T) != data_dest->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    rt::dag_build_guard build{rt::application::dag()};

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw invalid_parameter_error{"handler: explicit copy() is unsupported "
                                    "for queues not bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(dest);

    // Assume same element size and allocation shape as for dest
    rt::memory_location source_location{detail::get_host_device(), extract_ptr(src),
                                      rt::id<3>{}, data_dest->get_num_elements(),
                                      data_dest->get_element_size()};
    rt::memory_location dest_location{dev, rt::embed_in_id3(dest.get_offset()),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(dest.get_range()));

    rt::dag_node_ptr node = build.builder()->add_memcpy(
        std::move(explicit_copy), _requirements, _execution_hints);

    _command_group_nodes.push_back(node);
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


  const std::vector<rt::dag_node_ptr>& get_cg_nodes() const
  { return _command_group_nodes; }

  // defined in queue.hpp
  handler(const queue& q, async_handler handler, const rt::execution_hints& hints);

  const queue* _queue;
  detail::local_memory_allocator _local_mem_allocator;
  async_handler _handler;

  rt::requirements_list _requirements;
  rt::execution_hints _execution_hints;
  std::vector<rt::dag_node_ptr> _command_group_nodes;
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

}

} // namespace sycl
} // namespace hipsycl

#endif
