# C++ standard parallelism support

AdaptiveCpp supports automatic offloading of C++ standard algorithms.

## Installation & dependencies

C++ standard parallelism offload requires LLVM >= 15. It is automatically enabled when a sufficiently new LLVM is detected. Requires `cmake -DACPP_COMPILER_FEATURE_PROFILE=full` (this is the default setting) at cmake configure time.
C++ standard parallelism offload currently is only supported in conjunction with `libstdc++` >= 11. Other standard C++ standard library versions may or may not work. Support for `libc++` is likely easy to add if there is demand.

## Using accelerated C++ standard parallelism

Offloading of C++ standard parallelism is enabled using `--acpp-stdpar`. This flag does not by itself imply a target or compilation flow, which will have to be provided in addition using the normal `--acpp-targets` argument. C++ standard parallelism is expected to work with any of our clang compiler-based compilation flows, such as `omp.accelerated`, `cuda`, `hip` or the generic SSCP compiler (`--acpp-targets=generic`). It is not currently supported in library-only compilation flows. The focus of testing currently is the generic SSCP compiler.
AdaptiveCpp by default uses some experimental heuristics to determine if a problem is worth offloading. These heuristics are currently very simplistic and might not work well for you. They can be disabled using `--acpp-stdpar-unconditional-offload`.


## Algorithms and policies supported for offloading

Currently, the following execution policies qualify for offloading:

* `par_unseq`
* `par` (experimental; will only be offloaded on hardware that provides independent work item forward progress guarantees such as recent NVIDIA GPUs)

Offloading is implemented for the following STL algorithms:

| Algorithm | Notes |
|------------------|-------------------|
|`for_each`| |
|`for_each_n`| |
|`transform` | both unary and binary operator overloads |
|`copy` | |
|`copy_n` | |
|`copy_if` | |
|`move` | |
|`fill` | |
|`fill_n` | |
|`generate` | |
|`generate_n` | |
|`remove_copy` | |
|`remove_copy_if` | |
|`remove` | |
|`remove_if` | |
|`replace` | |
|`replace_if` | |
|`replace_copy` | |
|`replace_copy_if` | |
|`reverse` | |
|`reverse_copy` | |
|`transform_reduce` | all overloads |
|`reduce` | all overloads |
|`find` | |
|`find_if` | |
|`find_if_not` | |
|`find_end` | both overloads |
|`find_first_of` | both overloads |
|`any_of` | |
|`all_of` | |
|`none_of` | |
|`count` | |
|`count_if` | |
|`mismatch` | |
|`equal` | |
|`merge` | |
|`sort` | may not scale optimally for large problems |
|`min_element` | |
|`max_element` | |
|`is_sorted_until` | both overloads |
|`is_sorted` | both overloads |
|`inclusive_scan` | |
|`exclusive_scan` | |
|`transform_inclusive_scan` | |
|`transform_exclusive_scan` | |

For all other execution policies or algorithms, the algorithm will compile and execute correctly, however the regular host implementation of the algorithm provided by the C++ standard library implementation will be invoked and no offloading takes place.


## Performance

Performance can generally be expected to be on par with comparable SYCL kernels, although there are some optimizations specific to the C++ standard parallelism model. See the sections on execution and memory model below for details. However, because the implementation of C++ standard parallelism depends heavily on SYCL shared USM (unified shared memory) allocations, the implementation quality of USM at the driver and hardware level can have a great impact on performance, especially for memory-intensive applications.
In particular, on some AMD GPUs USM is known to not perform well due to hardware and driver limitations.
In general, USM relies on memory pages automatically migrating between host and device, depending on where they are accessed. Consequently, patterns where the same memory region is accessed by host and offloaded C++ standard algorithms in alternating fashion should be avoided as much as possible, as this will trigger memory transfers behind the scenes.

## Execution model

### Queues and devices

Each thread in the user application maintains a dedicated thread-local in-order SYCL queue that will be used to dispatch STL algorithms. Thus, concurrent operations can be expressed by launching them from separate threads.
The selected device is currently the device returned from the default selector. Use `ACPP_VISIBILITY_MASK` and/or backend-specific environment variables such as `HIP_VISIBLE_DEVICES` to control which device this is. Because `sycl::event` objects are not needed in the C++ standard parallelism model, queues are set up to rely exclusively on the hipSYCL coarse grained events extension. This means that offloading a C++ standard parallel algorithm can potentially have lower overhead compared to submitting a regular SYCL kernel.

### Synchronous and asynchronous execution

The C++ STL algorithms are all designed around the assumption of being synchronous. This can become a performance issue especially when multiple algorithms are executed in succession, as in principle a `wait()` must be executed after each algorithm is submitted to device.

To address this issue, a dedicated compiler optimization tries to remove `wait()` calls in between successive calls to offloaded algorithms, such that a `wait()` will only be executed for the last algorithm invocation. This is possible without side effects if no instructions (particularly loads and stores) between the algorithm invocations are present that might rely on memory changes initiated by the offload device.

This is an optimization, which has limits in terms of the depth of the analysis that the compiler performs. The following conditions can prevent synchronization elision, if they occur between two stdpar calls in the generated code:

* Calls to functions that are not inlined by the compiler, since this can prevent control flow analysis. Try to avoid any function calls in between stdpar calls (ideally also including `begin()` and `end()`);
* Calls to functions that are defined in other translation units, since the compiler then does not see the code that will be executed;
* Loads and stores, and other operations that may access or modify memory;
   * This can be mitigated to some extent by **not** using `--acpp-stdpar-system-usm`. Without system USM, AdaptiveCpp can assume that the stack is not accessible by offload devices, and therefore, any access to the stack does not prevent synchronization elision. With system USM however, this assumption no longer holds, and AdaptiveCpp needs to retain synchronization even for stack memory accesses! If you wish to rely on synchronization elision for performance, you might therefore want to **not** enable `--acpps-stdpar-system-usm`!
* Calling an stdpar algorithm like `transform_reduce` that returns the result of its computation as its return value - such an algorithm must always be synchronous due to its semantics!

In this example, the compiler might remove the synchronization after the `for_each()`:

```c++

auto first = data.begin();
auto last = data.end();
auto dest = dest.begin();
std::for_each(std::execution::par_unseq, first, last, ...);
std::transform(std::execution::par_unseq, first, last, dest, ...);

```


### Experimental: Automatic utilization of multiple devices using MQS (multi-queue scheduling)

AdaptiveCpp stdpar contains an **experimental** feature that allows it to schedule independent kernels to different queues on one device (which might improve device utilization), or even to different devices. This allows single-threaded stdpar programs to harness the power of multiple GPUs.

This works by compiler and runtime determining dependencies between kernels by investigating the accessed allocations. Independent kernels may then be assigned by the runtime to run on different queues or devices. For this decision, the runtime takes into account potential data transfer cost and the size of kernels.

In order for kernels to be scheduled to different devices or queues, several conditions must be met:

* The code must be compiled with `--acpp-targets=generic` and `--acpp-stdpar-mqs`.
* The compiler must be able to prove that the kernels exclusively access the allocations that are passed as kernel arguments (e.g. pointers captures in a lambda). In particular, the kernel must not perform *indirect access*, i.e. loading additional pointers from memory.
* In order to be able to run concurrently, kernels must access different sets of allocations, otherwise the compiler and runtime might assume a dependency.
* The synchronization elision optimization logic mentioned in the previous section must be able to elide the synchronization between the kernels. Please see the previous section for recommendations on how to ensure this.

If these conditions are met, the runtime can generate a task graph of the kernels which it then schedules across devices. **Note that currently, it can only schedule to multiple devices of the same backend.**

The following example illustrates a code pattern where the individual kernels for the different computational domains might be scheduled to different devices:

```c++

std::array<float*, NUM_DOMAINS> computational_domains = ... /* initialize domains somehow */

for(int i = 0; i < NUM_DOMAINS, ++i) {
    // These for_each calls are independent and might be scheduled to multiple
    // devices, or to different queues on a single device if only one device is
    // available.
    std::for_each(computational_domains[i], computational_domains[i]+domain_size, [=](auto& x){
        // operate on element from domain
    });
}

```

## Memory model

### Automatic migration of heap allocations to USM shared allocations

C++ is unaware of separate devices with their own device memory. In order to retain C++ semantics, when offloading C++ standard algorithms AdaptiveCpp tries to move all memory allocations that the application performs in translation units compiled with `--acpp-stdpar` to SYCL shared USM allocations. To this end, `operator new` and `operator delete` as well as the C-Style functions `malloc`, `aligned_alloc` and `free` are replaced by our own implementations (`calloc` and `realloc` are not yet implemented).
**Note that pointers to host stack memory cannot be used in offloaded C++ algorithms, because we cannot move stack allocations to USM memory! AdaptiveCpp detects whether such pointers are used, and will not offload in this case!**

This replacement is performed using a special compiler transformation. This compiler transformation also enforces that the SYCL headers perform regular allocations instead. This is important because in general the SYCL headers construct complex objects such as `std::vector` or `std::shared_ptr` which then get handed over to the SYCL runtime library. The runtime library however cannot rely on SYCL USM pointers -- in short: The runtime as the code responsible for managing these allocations cannot itself sit on them. Therefore, the compiler performs non-trivial operations to only selectively replace memory allocations.

The backend used to perform USM allocations is the backend managing the executing device as described in the previous section.


## Scope and visibility of replaced functions

Functions for memory allocation are only exchanged for USM variants within translation units compiled with `--acpp-stdpar`. Our USM functions for releasing memory are however overriding the standard functions within the entire application. This is motivated by the expectation that pointers may be shared within the application, and the place where they are released may not be the place where they are created. As our functions for freeing memory can handle both regular and USM allocations, making them more widely available seems like the safer choice.

Note that in C++ due to the one definition rule (ODR) the linker may in certain circumstances pick one definition of a symbol when multiple definitions are available. This can potentially be a problem if a user-defined function is both defined in a translation unit compiled with `--acpp-stdpar` and one without it. In this case, there is no guarantee that the linker will pick the variant that does USM allocations. Be aware that the most vulnerable code for this issue might not only be user code directly, but also header-only library code such as `std::` functions (think of e.g. the allocations performed by `std::vector` of common types) as these functions may be commonly used in multiple translation units.
**We therefore recommend that if you enable `--acpp-stdpar` for one translation unit, you also enable it for the other translation units in your project!**

Such issues are not present for the functions defined in the SYCL headers, because the compiler inserts special ABI tags into their symbol names when compiled with `--acpp-stdpar` to distinguish them from the regular variants, thus preventing such linking issues. Unfortunately, we cannot do the same for client code because we cannot know if other translation or linkage units will attempt to link against the user code, and expect the unaltered symbol names.

### User-controlled USM device pointers

Of course, if you wish to have greater control over memory, USM device pointers from user-controlled USM memory management function calls can also be used, as in any regular SYCL kernel. The buffer-accessor model is not supported; memory stored in `sycl::buffer` objects can only be used when converting it to a USM pointer using our buffer-USM interoperability extension.
Note that you may need to invoke SYCL functions to explicitly copy memory to device and back if you use explicit SYCL device USM allocations.
It is not recommended to use USM shared allocations from direct calls to `sycl::malloc_shared` in C++ standard algorithms. Such allocations will not be tracked by the stdpar runtime, and as such might not benefit from optimizations such as automatic prefetching.

### Systems with system-level USM support

If you are on a system that supports system-level USM, i.e. a system where every CPU pointer returned from regular memory allocations or even stack pointers can directly be used on GPUs (such as on AMD MI300 or Grace-Hopper), the compiler transformation to turn heap allocations to SYCL USM shared allocations is unnecessary. In this case, you may want to request the compiler to assume system-level USM and disable the compiler transformations regarding SYCL shared USM allocations using `--acpp-stdpar-system-usm`.

## Functionality supported in device code

The functionality supported in device code aligns with the kernel restrictions from SYCL. This means that no exceptions, dynamic polymorphism, dynamic memory management, or calls to external shared libraries are allowed. Note that this functionality might already be prohibited in the C++ `par_unseq` model anyway.

The `std::` math functions are supported in device code in an experimental state when using the generic SSCP compilation flow (`--acpp-targets=generic`). This is accomplished using a dedicated compiler pass that maps standard functions to our SSCP builtins.

When using the `par` execution policy, `std::atomic` and `std::atomic_ref` support in device code is available when using the generic SSCP compilation flow (`--acpp-targets=generic`), but experimental.
