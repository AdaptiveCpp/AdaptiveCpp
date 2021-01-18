# SYCL extensions in hipSYCL

hipSYCL implements several extensions that are not defined by the specification.

In general (and unless otherwise noted), hipSYCL extensions must be activated before they can be used by defining `HIPSYCL_EXT_<EXTENSIONNAME>` fore including `sycl.hpp`, for example:

```
#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>
```
Alternatively, instead of activating individual extensions, all extensions can be activated by defining `HIPSYCL_EXT_ENABLE_ALL`.

## Supported extensions


### `HIPSYCL_EXT_SCOPED_PARALLELISM`
This extension provides the scoped parallelism kernel invocation and programming model. This extension does not need to be enabled explicitly and is always available.
See [here](scoped-parallelism.md) for more details. **Scoped parallelism is the recommended way in hipSYCL to write programs that are performance portable between CPU and GPU backends.**

### `HIPSYCL_EXT_ENQUEUE_CUSTOM_OPERATION`

This extension allows to enqueue custom device operations for efficient interoperability with backends. This extension does not need to be enabled explicitly and is always available.
See [here](enqueue-custom-operation.md) for more details.

### `HIPSYCL_EXT_BUFFER_USM_INTEROP`

hipSYCL supports interoperability between `sycl::buffer` and USM pointers. See [here](buffer-usm-interop.md) for details.

### HIPSYCL_EXT_CG_PROPERTY_*: Command group properties

hipSYCL supports attaching special command group properties to individual command groups. This is done by passing a property list to the queue's `submit` member function:

```cpp
template <typename T>
event queue::submit(const property_list& prop_list, T cgf)
```

A list of supported command group properties follows:

#### HIPSYCL_EXT_CG_PROPERTY_PREFER_GROUP_SIZE

##### API reference

```cpp
namespace sycl::property::command_group {

template<int Dim>
struct hipSYCL_prefer_group_size {
  hipSYCL_prefer_group_size(range<Dim> group_size);
};

}
```

##### Description

If this property is added to a command group property list, it instructs the backend to prefer a particular work group size for kernels for models where a SYCL implementation has the freedom to decide on a work group size.

In the current implementation, this property only affects the selected local size for basic parallel for on HIP and CUDA backends.

*Note:* The property only affects kernel launches of the same dimension. If you want to set the group size for 2D kernels, you need to attach a `hipSYCL_prefer_group_size<2>` property.

#### HIPSYCL_EXT_CG_PROPERTY_RETARGET

##### API reference

```cpp
namespace sycl::property::command_group {

template<int Dim>
struct hipSYCL_retarget {
  hipSYCL_retarget(const device& dev);
};

}
```

##### Description

If this property is added to a command group property list, it instructs the hipSYCL runtime to execute the submitted operation on a different device than the one the queue was bound to. This can be useful as a more convenient mechanism to dispatch to multiple devices compared to creating multiple queues.

Using this property does *not* introduce additional overheads compared to using multiple queues. In particular, it does *not* silently lead to the creation of additional backend execution resources such as CUDA streams.

In order to understand this, it is important to realize that because of the design of the hipSYCL runtime, a queue is decoupled from backend objects. Instead, the hipSYCL runtime internally manages a pool of backend execution resources such as CUDA streams, and automatically distributes work across those resources.
In this design, a queue is nothing more than an interface to hipSYCL runtime functionality. This allows us to efficiently retarget operations submitted to a queue arbitrarily.

Compared to using multiple queues bound to different devices, using a single queue and submitting with the `hipSYCL_retarget` property comes with some minor semantic differences:

* A single `queue::wait()` call guarantees that all operations submitted to the queue, no matter to which device they were retargeted, have completed. With multiple queues on the other hand, multiple `wait()` calls are necessary which can add some overhead.
* If the queue is an in-order queue, the in-order property is *preserved even if the operations are retargeted to run on different devices*. This can be a highly convenient way to formulate in-order USM algorithms that require processing steps on different devices.

### `HIPSYCL_EXT_FP_ATOMICS`
This extension allows atomic operations on floating point types. Since this is not in the spec, this may break portability. Additionally, not all hipSYCL backends may support the same set of FP atomics. It is the user's responsibility to ensure that the code remains portable and to implement fallbacks for platforms that don't support this.

### `HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE`
This SYCL extension allows to `require()` placeholder accessors automatically. This extension does not need to be enabled explicitly and is always available.

The following example illustrates the use of this extension:

```cpp

cl::sycl::queue q;
cl::sycl::buffer<int, 1> buff{/* Initialize somehow */};
cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, 
      cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc{buff};

// This will call handler::require(acc) for each command group subsequently
// launched in the queue, until the `automatic_requirement` object is destroyed
// or `automatic_requirement.release()` is called.
auto automatic_requirement = 
   cl::sycl::vendor::hipsycl::automatic_require(q, acc);
// The member function `is_required()` can be used to check if
// if the automatic requirement object is active:
assert(automatic_requirement.is_required());

q.submit([&] (cl::sycl::handler& cgh) {
  // No call to require() is necessary here!
  cgh.parallel_for<class kernel1>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

q.submit([&] (cl::sycl::handler& cgh) {
  cgh.parallel_for<class kernel2>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

// After release(), acc will not be required anymore by kernels submitted to the queue
// afterwards.
automatic_requirement.release();
assert(!automatic_requirement.is_required());

/* Kernel submissions that do not need acc could go here */

// acc will now be required again
automatic_requirement.reacquire();

q.submit([&] (cl::sycl::handler& cgh) {
  cgh.parallel_for<class kernel3>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

// If the automatic_requirement object goes out of scope, it will release the auto requirement
// if it is active.
```

This extension serves two purposes:
1. Avoid having to call `require()` again and again if the same accessor is used in many subsequent kernels. This can lead to a significant reduction of boilerplate code.
2. Simplify code when working with SYCL libraries that accept lambda functions or function objects. For example, for a `sort()` function in a SYCL library a custom comparator may be desired. Currently, there is no easy way to access some independent data in that comparator because accessors must be requested in the command group handler. This would not be possible in that case since the command group would be spawned internally by `sort`, and the user has no means of accessing it.

## `HIPSYCL_EXT_CUSTOM_PFWI_SYNCHRONIZATION`
This extension allows for the user to specify what/if synchronization should happen at the end of a `parallel_for_work_item` call.
This extension is always enabled and does not need to be enabled explicitly.

Example:
```cpp
// By default, a barrier() will be executed at the end of
// a parallel for work item call, as defined by the spec and shown here:
group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
  ...
});
// The extension allows the user to specify custom 'finalizers' to
// alter synchronization behavior:
namespace sync = cl::sycl::vendor::hipsycl::synchronization;
group.parallel_for_work_item<sync::none>(
  [&](cl::sycl::h_item<1> item) {
  // No synchronization will be done after this call
});

group.parallel_for_work_item<sync::local_mem_fence>(
  [&](cl::sycl::h_item<1> item) {
  // local mem_fence will be done after this call
});
```
The following 'finalizers' are supported:
* `cl::sycl::vendor::hipsycl::synchronization::none` - no Operation
* `cl::sycl::vendor::hipsycl::synchronization::barrier<access::fence_space>` - barrier
* `cl::sycl::vendor::hipsycl::synchronization::local_barrier` - same as `barrier<access::fence_space::local_space>`
* `cl::sycl::vendor::hipsycl::synchronization::mem_fence<access::fence_space, access::mode = access::mode::read_write>` - memory fence
* `cl::sycl::vendor::hipsycl::synchronization::local_mem_fence` - same as `mem_fence<access::fence_space::local_space>`
* `cl::sycl::vendor::hipsycl::synchronization::global_mem_fence` - same as `mem_fence<access::fence_space::global_space>`
* `cl::sycl::vendor::hipsycl::synchronization::global_and_local_mem_fence` - same as `mem_fence<access::fence_space::global_and_local>`
