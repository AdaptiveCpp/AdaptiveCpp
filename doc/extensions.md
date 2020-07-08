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



