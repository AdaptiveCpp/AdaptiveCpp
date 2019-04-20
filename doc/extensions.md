# SYCL extensions in hipSYCL

hipSYCL implements several extensions that are not defined by the specification.

In general (and unless otherwise noted), hipSYCL extensions must be activated before they can be used by defining `HIPSYCL_EXT_<EXTENSIONNAME>` fore including `sycl.hpp`, for example:

```
#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>
```
Alternatively, instead of activating individual extensions, all extensions can be activated by defining `HIPSYCL_EXT_ENABLE_ALL`.

## Supported extensions

### `HIPSYCL_EXT_FP_ATOMICS`
This extension allows atomic operations on floating point types. Since this is not in the spec, this may break portability. Additionally, not all hipSYCL backends may support the same set of FP atomics. It is the user's responsibility to ensure that the code remains portable and to implement fallbacks for platforms that don't support this.

### `HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE`
This SYCL extension allows to `require()` placeholder accessors automatically, as seen in the following example:

```cpp

cl::sycl::queue q;
cl::sycl::buffer<int, 1> buff{/* Initialize somehow */};
cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, 
      cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc{buff};

// This will call handler::require(acc) for each command group subsequently
// launched in the queue, until the `automatic_requirement` object is destroyed
// or `automatic_requirement.release()` is called.
auto automatic_requirement = cl::sycl::automatic_require(q, acc);
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
