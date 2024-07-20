/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_HANDLER_HPP
#define HIPSYCL_HANDLER_HPP

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "exception.hpp"
#include "access.hpp"
#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "libkernel/sscp/builtins/print.hpp"
#include "types.hpp"
#include "usm_query.hpp"
#include "libkernel/accessor.hpp"
#include "libkernel/backend.hpp"
#include "libkernel/builtin_kernels.hpp"
#include "libkernel/id.hpp"
#include "libkernel/range.hpp"
#include "libkernel/reduction.hpp"
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
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/glue/embedded_pointer.hpp"
#include "hipSYCL/glue/kernel_launcher_factory.hpp"
#include "hipSYCL/glue/kernel_names.hpp"
#include "hipSYCL/glue/generic/code_object.hpp"

#include "hipSYCL/algorithms/reduction/reduction_engine.hpp"
#include "hipSYCL/algorithms/util/memory_streaming.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

#ifndef ACPP_FORCE_INSTANT_SUBMISSION
#define ACPP_FORCE_INSTANT_SUBMISSION 0
#endif

#if defined(HIPSYCL_ALLOW_INSTANT_SUBMISSION) && !defined(ACPP_ALLOW_INSTANT_SUBMISSION)
#define ACPP_ALLOW_INSTANT_SUBMISSION HIPSYCL_ALLOW_INSTANT_SUBMISSION
#endif

#if ACPP_FORCE_INSTANT_SUBMISSION
#undef ACPP_ACPP_ALLOW_INSTANT_SUBMISSION
#define ACPP_ACPP_ALLOW_INSTANT_SUBMISSION 1
#endif

#ifndef ACPP_ALLOW_INSTANT_SUBMISSION
#define ACPP_ALLOW_INSTANT_SUBMISSION 0
#endif

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
      throw exception{make_error_code(errc::invalid),
                      "handler: require(): accessor is illegal paramater for "
                      "require() because it is not bound to a buffer."};
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
    throw exception{make_error_code(errc::invalid),
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
      throw exception{make_error_code(errc::invalid),
                      "handler: require(): accessor is illegal paramater for "
                      "require() because it is not bound to a buffer."};
    }

    // get_offset and get_range are only defined for dimensions > 0
    id<dimensions == 0 ? 1 : dimensions> offset;
    if constexpr (dimensions == 0)
      offset = id<1>(0);
    else
      offset = acc.get_offset();

    range<dimensions == 0 ? 1 : dimensions> range;
    if constexpr (dimensions == 0)
      range = range<1>(1);
    else
      range = acc.get_range();

    detail::accessor_data<dimensions == 0 ? 1 : dimensions> data{
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


  template <typename KernelName = __acpp_unnamed_kernel, typename KernelType>
  void single_task(KernelType kernelFunc)
  {
    this->submit_kernel<KernelName, rt::kernel_type::single_task>(
      sycl::id<1>{0}, sycl::range<1>{1}, sycl::range<1>{1}, kernelFunc);
  }

  template <typename KernelName = __acpp_unnamed_kernel,
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

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel>
  void parallel_for(range<1> numWorkItems,
                    const ReductionsAndKernel &... redu_kernel) {

    auto invoker = [&](auto&& kernel, auto&&... reductions){
      this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
          sycl::id<1>{}, numWorkItems,
          get_preferred_group_size<1>(),
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  template <typename KernelName = __acpp_unnamed_kernel,
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

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel>
  void parallel_for(range<1> numWorkItems,
                    id<1> workItemOffset,
                    const ReductionsAndKernel &... redu_kernel) {
    auto invoker = [&](auto&& kernel, auto&& ... reductions) {
      this->submit_kernel<KernelName, rt::kernel_type::basic_parallel_for>(
          workItemOffset, numWorkItems,
          get_preferred_group_size<1>(),
          kernel, reductions...);
    };

    detail::separate_last_argument_and_apply(invoker, redu_kernel...);
  }

  template <typename KernelName = __acpp_unnamed_kernel,
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
  template <typename KernelName= __acpp_unnamed_kernel,
            typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               WorkgroupFunctionType kernelFunc)
  {
    dispatch_hierarchical_kernel(numWorkGroups,
                                 get_default_local_range<dimensions>(),
                                 kernelFunc);
  }
  */

  template <typename KernelName = __acpp_unnamed_kernel,
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
  
  template <typename KernelName = __acpp_unnamed_kernel,
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
        throw exception{make_error_code(errc::invalid),
                        "handler: copy(): Accessor sizes are incompatible."};
      }
    }

    std::shared_ptr<rt::buffer_data_region> data_src  = get_memory_region(src);
    std::shared_ptr<rt::buffer_data_region> data_dest = get_memory_region(dest);

    if (sizeof(T) != data_src->get_element_size() ||
        sizeof(T) != data_dest->get_element_size())
      assert(false && "Accessors with different element size than original "
                      "buffer are not yet supported");

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit copy() is unsupported for queues not "
                      "bound to devices"};

    rt::device_id src_dev = get_explicit_accessor_target(src);
    rt::device_id dest_dev = get_explicit_accessor_target(dest);
    
    rt::memory_location source_location{src_dev, rt::embed_in_id3(get_offset(src)),
                                        data_src};
    rt::memory_location dest_location{dest_dev, rt::embed_in_id3(get_offset(dest)),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(src)));

    rt::dag_node_ptr node = create_task(std::move(explicit_copy), _execution_hints);

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
      throw exception{make_error_code(errc::invalid),
                      "handler: device update() is unsupported for queues not "
                      "bound to devices"};

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


    this->submit_kernel<__acpp_unnamed_kernel, rt::kernel_type::basic_parallel_for>(
        sycl::id<dim>{}, get_range(dest),
        get_preferred_group_size<dim>(),
        detail::kernels::fill_kernel{dest, src});
  }

  // ------ USM functions ------

  void memcpy(void *dest, const void *src, std::size_t num_bytes) {

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit memcpy() is unsupported for queues "
                      "not bound to devices"};

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

      throw exception{make_error_code(errc::invalid),
                      "Invalid allocation type"};
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

    rt::dag_node_ptr node = create_task(std::move(op), _execution_hints);

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
        throw exception{make_error_code(errc::invalid),
                        "handler: USM fill() is unsupported for queues not "
                        "bound to devices"};

      this->submit_kernel<__acpp_unnamed_kernel,
                          rt::kernel_type::basic_parallel_for>(
          sycl::id<1>{}, sycl::range<1>{count},
          get_preferred_group_size<1>(),
          detail::kernels::fill_kernel_usm{typed_ptr, pattern});
    }
  }

  void memset(void *ptr, int value, std::size_t num_bytes) {

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit memset() is unsupported for queues "
                      "not bound to devices"};

    auto op = rt::make_operation<rt::memset_operation>(
        ptr, static_cast<unsigned char>(value), num_bytes);

    rt::dag_node_ptr node = create_task(std::move(op), _execution_hints);

    _command_group_nodes.push_back(node);
  }

  void prefetch_host(const void *ptr, std::size_t num_bytes) {

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit prefetch() is unsupported for queues "
                      "not bound to devices"};

    rt::device_id executing_dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();

    // The device to which we prefetch
    rt::device_id target_dev = detail::get_host_device();

    // If this queue is on the host, we need to execute on the backend
    // that has provided the USM pointer instead.
    rt::execution_hints hints = _execution_hints;

    if (executing_dev.is_host()) {

      auto usm_dev = detail::extract_rt_device(get_pointer_device(ptr, _ctx));

      hints.set_hint(rt::hints::bind_to_device{
          usm_dev});
    }

    auto op = rt::make_operation<rt::prefetch_operation>(
        ptr, num_bytes, target_dev);

    rt::dag_node_ptr node = create_task(std::move(op), hints);

    _command_group_nodes.push_back(node);
  }

  void prefetch(const void *ptr, std::size_t num_bytes) {

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit prefetch() is unsupported for queues "
                      "not bound to devices"};

    rt::device_id executing_dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();

    if (executing_dev.is_host())
      // If this queue is bound to a host device, run prefetch_host()
      // to potentially get data back to host.
      prefetch_host(ptr, num_bytes);
    else {
      // Otherwise, run prefetch on the queue's device to the
      // queue's device

      auto op = rt::make_operation<rt::prefetch_operation>(
          ptr, num_bytes, executing_dev);

      rt::dag_node_ptr node = create_task(std::move(op), _execution_hints);

      _command_group_nodes.push_back(node);
    }
  }

  void mem_advise(const void *addr, std::size_t num_bytes, int advice) {
    throw exception{make_error_code(errc::feature_not_supported),
                    "mem_advise() is not yet supported"};
  }


  template <class InteropFunction>
  void AdaptiveCpp_enqueue_custom_operation(InteropFunction f) {
    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: submitting custom operations is unsupported "
                      "for queues not bound to devices"};

    auto custom_kernel_op = rt::make_operation<rt::kernel_operation>(
        typeid(f).name(),
        glue::make_kernel_launcher<class _unnamed, rt::kernel_type::custom>(
            sycl::id<3>{}, sycl::range<3>{}, 
            sycl::range<3>{},
            0, f),
        _requirements);

    rt::dag_node_ptr node = create_task(std::move(custom_kernel_op), _execution_hints);
    
    _command_group_nodes.push_back(node);
  }

  template<class InteropFunction>
  [[deprecated("Use AdaptiveCpp_enqueue_custom_operation()")]]
  void hipSYCL_enqueue_custom_operation(InteropFunction f) {
    AdaptiveCpp_enqueue_custom_operation(f);
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
      throw exception{make_error_code(errc::invalid),
                      "update_dev(): Accessor is not bound to buffer"};

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

    // Merge new hint into default hints
    rt::execution_hints hints = _execution_hints;
    hints.set_hint(rt::hints::bind_to_device{
            dev});
    assert(hints.has_hint<rt::hints::bind_to_device>());
    assert(hints.get_hint<rt::hints::bind_to_device>()->get_device_id() ==
           dev);

    rt::dag_node_ptr node = create_task(std::move(explicit_requirement), hints);

    _command_group_nodes.push_back(node);
  }

  template <class KernelName, class KernelFuncType, int Dim,
            typename... Reductions>
  rt::dag_node_ptr submit_ndrange_reduction_kernel(
      sycl::range<Dim> global_range, sycl::range<Dim> local_range,
      KernelFuncType f, Reductions... reductions) {

    const std::size_t local_size = local_range.size();

    rt::dag_node_ptr previous_event;
    auto ndrange_launcher = [&](std::size_t num_groups, std::size_t wg_size,
                                std::size_t global_size, std::size_t local_mem,
                                auto kernel) {
      rt::requirements_list req_list{_rt};
      
      if(previous_event)
        req_list.add_node_requirement(previous_event);
      // Also need to add existing memory requirements, so that
      // additional kernels will also create dependencies on
      // buffers for buffer-accessor reductions
      for(const rt::dag_node_ptr& req : _requirements.get()) {
        auto* op = req->get_operation();
        if(op->is_requirement()) {
          auto cloned_op =
              static_cast<rt::requirement *>(op)->clone_requirement(true);

          req_list.add_requirement(std::move(cloned_op));
        } else {
          // Other dependencies that are not requirements should be
          // covered by the dependency to the previous node that we add
          // before this for loop.
        }
      }
      
      previous_event =
          this->submit_kernel_impl<__acpp_unnamed_kernel,
                                   rt::kernel_type::ndrange_parallel_for>(
              {}, sycl::range{num_groups * wg_size}, sycl::range{wg_size},
              kernel, local_mem, req_list);
    };

    using group_reduction_type =
        algorithms::reduction::wg_model::group_reductions::generic_local_memory<
            std::decay_t<decltype(reductions)>...>;
    
    // The reduction engine will update this value with the
    // appropriate amount of local memory for the main kernel.
    std::size_t main_kernel_local_mem = _local_mem_allocator.get_allocation_size();
    rt::device_id dev =
        _execution_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();

    algorithms::util::allocation_group scratch_allocations {
        _allocation_cache, dev};

    algorithms::reduction::wg_model::group_horizontal_reducer<
        group_reduction_type>
        horizontal_reducer{
            group_reduction_type{main_kernel_local_mem, local_size}};

    algorithms::reduction::wg_hierarchical_reduction_engine engine{
        horizontal_reducer, &scratch_allocations};

    const std::size_t dispatched_global_size = global_range.size();

    auto plan = engine.create_plan(dispatched_global_size, local_size,
                                  reductions...);
    
    auto generate_sycl_reducer = [](auto& wg_model_reducer) {
      using reducer_t = std::decay_t<decltype(wg_model_reducer)>;

      return reducer<reducer_t>{wg_model_reducer};
    };

    auto make_lvalue_reducers = [](auto func, auto idx, auto... reducers){
      func(idx, reducers...);
    };

    auto main_kernel = engine.make_main_reducing_kernel(
        [=](sycl::nd_item<Dim> idx, auto &...reducers) {
          make_lvalue_reducers(f, idx, generate_sycl_reducer(reducers)...);
        },
        plan);

    // (For non-inorder queue) main kernel needs dependency
    // on most recent preceding reduction last event
    if(_most_recent_reduction_kernel) {
      if(auto node = _most_recent_reduction_kernel->lock()) {
        _requirements.add_node_requirement(node);
      }
    }
    previous_event =
        this->submit_kernel_impl<__acpp_unnamed_kernel,
                                 rt::kernel_type::ndrange_parallel_for>(
            {}, global_range, local_range, main_kernel, main_kernel_local_mem,
            _requirements);

    engine.run_additional_kernels(ndrange_launcher, plan);

    return previous_event;
  }

  // Kernel submission with reductions
  template <class KernelName, rt::kernel_type KernelType, class KernelFuncType,
            int Dim, typename... Reductions>
  void submit_kernel(sycl::id<Dim> offset, sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range, KernelFuncType f,
                     Reductions... reductions) {
    static_assert(sizeof...(reductions) > 0,
                  "Overload resolution should never pick this overload without "
                  "reductions");

    if constexpr(KernelType == rt::kernel_type::ndrange_parallel_for) {
      _command_group_nodes.push_back(
          submit_ndrange_reduction_kernel<KernelName>(global_range, local_range,
                                                      f, reductions...));
    }
    else if constexpr(KernelType == rt::kernel_type::basic_parallel_for) {
      auto default_local_range = [](){
        if constexpr(Dim == 3)
          return sycl::range<3>{1,1,128};
        else if constexpr(Dim == 2)
          return sycl::range<2>{1,128};
        else
          return sycl::range<1>{128};
      };

      local_range = default_local_range();

      bool can_use_data_streamer = 
        (Dim == 1) && _execution_hints.has_hint<rt::hints::bind_to_device>();

      if (can_use_data_streamer) {
        const std::size_t desired_global_range = global_range.size();

        algorithms::util::data_streamer streamer{
            _execution_hints.get_hint<rt::hints::bind_to_device>()
              ->get_device_id(), desired_global_range, local_range.size()};
        std::size_t dispatched_global_range = streamer.get_required_global_size();

        auto wrapped_f = [desired_global_range, f](sycl::nd_item<1> idx, auto &...reducers) {
          algorithms::util::data_streamer::run(
              desired_global_range, idx, [&](sycl::id<1> i) {
                auto this_item = sycl::detail::make_item<1>(
                    i, sycl::range{desired_global_range});
                if constexpr (Dim == 1) {
                  f(this_item, reducers...);
                }
              });
        };

        // Ensure everything is submitted as 1D since data streaming
        // can only work for 1D kernels.
        _command_group_nodes.push_back(
            submit_ndrange_reduction_kernel<KernelName>(
                sycl::range<1>{dispatched_global_range},
                sycl::range<1>{local_range.size()}, wrapped_f, reductions...));
      } else {
        auto wrapped_f = [=](sycl::nd_item<Dim> idx, auto &...reducers) {
          auto gid = idx.get_global_id();
          auto this_item = sycl::detail::make_item<Dim>(gid, global_range);

          for (int i = 0; i < Dim; ++i)
            if (gid[i] >= global_range[i])
              return;
          f(this_item, reducers...);
        };

        sycl::range<Dim> num_groups;
        for(int i = 0; i < Dim; ++i)
          num_groups[i] = (global_range[i] + local_range[i] - 1)/local_range[i];

        _command_group_nodes.push_back(
            submit_ndrange_reduction_kernel<KernelName>(
                num_groups * local_range, local_range, wrapped_f,
                reductions...));
      }
    }
  }

  // Plain kernel submission without reductions
  template <class KernelName, rt::kernel_type KernelType, class KernelFuncType,
            int Dim>
  void submit_kernel(sycl::id<Dim> offset, sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range, KernelFuncType f) {

    std::size_t local_mem_size = _local_mem_allocator.get_allocation_size();
    rt::dag_node_ptr node = submit_kernel_impl<KernelName, KernelType>(
        offset, global_range, local_range, f, local_mem_size, _requirements);
    _command_group_nodes.push_back(node);
  }

  template <class KernelName, rt::kernel_type KernelType, class KernelFuncType,
            int Dim>
  rt::dag_node_ptr
  submit_kernel_impl(sycl::id<Dim> offset, sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range, KernelFuncType f,
                     std::size_t local_mem_size,
                     const rt::requirements_list &req_list) {

    auto kernel_op = rt::make_operation<rt::kernel_operation>(
        typeid(KernelFuncType).name(),
        glue::make_kernel_launcher<KernelName, KernelType>(
            offset, local_range, global_range, local_mem_size, f),
        _requirements);

    rt::dag_node_ptr node =
        create_task(std::move(kernel_op), _execution_hints, req_list);

    // This registers the kernel with the runtime when the application
    // launches, and allows us to introspect available kernels.
    ACPP_STATIC_KERNEL_REGISTRATION(KernelFuncType);

    return node;
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

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit copy() is unsupported for queues not "
                      "bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(src);

    rt::memory_location source_location{dev, rt::embed_in_id3(get_offset(src)),
                                        data_src};
    // Assume the allocation behind dest is large enough to hold src.get_range.size() contiguous elements
    rt::memory_location dest_location{detail::get_host_device(), extract_ptr(dest),
                                      rt::id<3>{}, rt::embed_in_range3(get_range(src)),
                                      data_src->get_element_size()};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(src)));

    rt::dag_node_ptr node = create_task(std::move(explicit_copy), _execution_hints);

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

    if(!_execution_hints.has_hint<rt::hints::bind_to_device>())
      throw exception{make_error_code(errc::invalid),
                      "handler: explicit copy() is unsupported for queues not "
                      "bound to devices"};

    rt::device_id dev = get_explicit_accessor_target(dest);

    // Assume src contains src.get_range.size() contiguous elements
    rt::memory_location source_location{detail::get_host_device(), extract_ptr(src),
                                      rt::id<3>{}, rt::embed_in_range3(get_range(dest)),
                                      data_dest->get_element_size()};
    rt::memory_location dest_location{dev, rt::embed_in_id3(get_offset(dest)),
                                      data_dest};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, rt::embed_in_range3(get_range(dest)));

    rt::dag_node_ptr node = create_task(std::move(explicit_copy), _execution_hints);

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

  const rt::node_list_t& get_cg_nodes() const
  { return _command_group_nodes; }

  bool contains_non_instant_nodes() const {
    return _contains_non_instant_nodes;
  }

  
  handler(const context &ctx, async_handler handler,
          const rt::execution_hints &hints, rt::runtime* rt,
          algorithms::util::allocation_cache* cache,
          std::weak_ptr<rt::dag_node>* most_recent_reduction_kernel)
      : _ctx{ctx}, _handler{handler}, _execution_hints{hints},
        _preferred_group_size1d{}, _preferred_group_size2d{},
        _preferred_group_size3d{}, _rt{rt}, _requirements{rt},
        _allocation_cache{cache},
        _most_recent_reduction_kernel{most_recent_reduction_kernel}{}

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


  rt::dag_node_ptr create_task(std::unique_ptr<rt::operation> op,
                               const rt::execution_hints &hints,
                               const rt::requirements_list& requirements) {

    bool uses_buffers = false;
    bool has_non_instant_dependency = false;
    bool is_unbound = !hints.has_hint<rt::hints::bind_to_device>();

    for(const auto& req : requirements.get()) {
      if(req->get_operation()->is_requirement())
        uses_buffers = true;
      if (!req->get_execution_hints()
               .has_hint<rt::hints::instant_execution>() &&
          !req->is_known_complete())
        has_non_instant_dependency = true;
    }
    
    bool is_dedicated_in_order_queue = false;
    rt::backend_executor* executor = nullptr;
    if(hints.has_hint<rt::hints::prefer_executor>()) {
      executor = hints.get_hint<rt::hints::prefer_executor>()->get_executor();
    }
    if(executor && executor->is_inorder_queue())
      is_dedicated_in_order_queue = true;

    if (!ACPP_ALLOW_INSTANT_SUBMISSION || uses_buffers ||
        has_non_instant_dependency || is_unbound ||
        !is_dedicated_in_order_queue ||
        op->is_requirement()) {
#if ACPP_FORCE_INSTANT_SUBMISSION
      throw exception{make_error_code(errc::invalid), "Instant submission not possible, "
          "but application was built with ACPP_FORCE_INSTANT_SUBMISSION=1"};
#else
      // traditional submission
      rt::dag_build_guard build{_rt->dag()};
      _contains_non_instant_nodes = true;

      return build.builder()->add_command_group(std::move(op), requirements, hints);
#endif
    } else {

      rt::dag_node_ptr node = std::make_shared<rt::dag_node>(
          hints, requirements.get(), std::move(op), _rt);
      node->assign_to_device(
          hints.get_hint<rt::hints::bind_to_device>()->get_device_id());
      node->assign_to_executor(executor);
      // Remember this was instant submission
      node->get_execution_hints().set_hint(rt::hints::instant_execution{});

      executor->submit_directly(node, node->get_operation(), requirements.get());
      // Signal that instrumentation setup phase is complete
      node->get_operation()->get_instrumentations().mark_set_complete();
      return node;
    }
  }

  rt::dag_node_ptr create_task(std::unique_ptr<rt::operation> op,
                               const rt::execution_hints &hints) {
    return create_task(std::move(op), hints, _requirements);
  }

  const context _ctx;
  detail::local_memory_allocator _local_mem_allocator;
  async_handler _handler;

  rt::requirements_list _requirements;
  const rt::execution_hints& _execution_hints;
  rt::node_list_t _command_group_nodes;

  range<1> _preferred_group_size1d;
  range<2> _preferred_group_size2d;
  range<3> _preferred_group_size3d;

  rt::runtime* _rt;

  bool _contains_non_instant_nodes = false;

  algorithms::util::allocation_cache* _allocation_cache;

  std::weak_ptr<rt::dag_node>* _most_recent_reduction_kernel;
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
