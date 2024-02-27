# AdaptiveCpp performance guide


AdaptiveCpp has been repeatedly shown to deliver very competitive performance compared to other SYCL implementations or proprietary solutions like CUDA. See for example:

* *Sohan Lal, Aksel Alpay, Philip Salzmann, Biagio Cosenza, Nicolai Stawinoga, Peter Thoman, Thomas Fahringer, and Vincent Heuveline. 2020. SYCL-Bench: A Versatile Single-Source Benchmark Suite for Heterogeneous Computing. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 10, 1. DOI:https://doi.org/10.1145/3388333.3388669*
* *Brian Homerding and John Tramm. 2020. Evaluating the Performance of the hipSYCL Toolchain for HPC Kernels on NVIDIA V100 GPUs. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 16, 1–7. DOI:https://doi.org/10.1145/3388333.3388660*
* *Tom Deakin and Simon McIntosh-Smith. 2020. Evaluating the performance of HPC-style SYCL applications. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 12, 1–11. DOI:https://doi.org/10.1145/3388333.3388643*


## LLVM stack

* Building AdaptiveCpp against newer LLVM generally results in better performance for backends that are relying on LLVM.
* Note that when comparing AdaptiveCpp against other LLVM-based compilers, the comparison is usually only really fair if both are based on the same LLVM version.

## Use the right compilation flow: Generic is typically fastest

* Despite its name, **generic compilation does not imply that the compiler compiles to a less target-specific and thus less performant code representation**.
* The `generic` compiler instead relies on runtime JIT compilation & optimization, i.e. it optimizes when it sees the exact target hardware at runtime.
* `generic` also implements many optimizations that are not available in the other compilers.

`generic` always compiles *faster*, usually produces *faster* binaries and creates binaries that are more *portable* as they can directly offload to all supported devices.

The other compilation flows `omp`, `cuda`, `hip` should mainly be used when *interoperability* with vendor programming languages like CUDA or HIP is more important than benefitting from the latest compiler features.

## Ahead-of-time vs JIT compilation

The compilation targets `omp`, `hip` and `cuda` perform ahead-of-time compilation. This means they depend strongly on the user to provide correct optimization flags when compiling.
You should thus always compile with e.g. `-O3`, and additionally `-march=native` or comparable for the CPU backend `omp`.

The `generic` target on the other hand relies on JIT compilation at runtime, and mainly optimizes kernels at runtime. Its kernel performance is less sensitive to user-provided optimization flags.
However, `generic` has a slight overhead the first time it launches a kernel since it carries out JIT compilation at that point.
For future application runs, this initial overhead is reduced as it leverages an on-disk persistent kernel cache.

## Generic target

### Adaptivity

The generic compilation target will try to create increasingly more optimized kernels based on information gathered at runtime. This can allow for faster kernels in future application runs. Because of this, to obtain peak performance, an application may have to be run multiple times.

This optimization process is complete when the following warning is no longer printed:
```
[AdaptiveCpp Warning] kernel_cache: This application run has resulted in new "binaries being JIT-compiled. This indicates that the runtime optimization process has not yet reached peak performance. You may want to run the application again until this warning no longer appears to achieve optimal performance.
```

The extent of this can be controlled using the environment variable `ACPP_ADAPTIITY_LEVEL`. A value of 0 disables the feature. The default is 1. Higher levels are expected to result in higher peak performance, but may require more application runs to converge to this performance. The default level of 1 usually guarantees peak performance for the second application run.

**For peak performance, you should not disable adaptivity, and run the application until the warning above is no longer printed.**

*Note: Adaptivity levels higher than 1 are currently not implemented.*

### Empty the kernel cache when upgrading the stack

The generic compiler also relies on an on-disk persistent kernel cache to speed up kernel JIT compilation. This cache usually resides in `$HOME/.acpp/apps`.
If you have made any changes to the stack that the AdaptiveCpp runtime is not aware of (e.g. upgrade AdaptiveCpp itself, or other lower-level components of the stack like drivers), you may want to force recompilation of kernels. Otherwise it might still use the old kernels from the cache, and you may thus not benefit e.g. from compiler upgrades.
Clearing the cache can be accomplished by simply clearing the cache directory, e.g. `rm -rf ~/.acpp/apps/*`

*Note: Changes in user application code do not require clearing the kernel cache.*

## CPU backend

* Enable OpenMP thread pinning (e.g. `OMP_PROC_BIND=true`). AdaptiveCpp uses asynchronous worker threads for some light-weight tasks such as garbage collection, and these additional threads can interfere with kernel execution if OpenMP threads are not bound to cores.

### With omp.* compilation flow
* When using `OMP_PROC_BIND`, there have been observations that performance suffers substantially, if AdaptiveCpp's OpenMP backend has been compiled against a different OpenMP implementation than the one used by `acpp` under the hood. For example, if `omp.acclerated` is used, `acpp` relies on clang and typically LLVM `libomp`, while the AdaptiveCpp runtime library may have been compiled with gcc and `libgomp`. The easiest way to resolve this is to appropriately use `cmake -DCMAKE_CXX_COMPILER=...` when building AdaptiveCpp to ensure that it is built using the same compiler. **If you oberve substantial performance differences between AdaptiveCpp and native OpenMP, chances are your setup is broken.**

### With omp.library-only compilation flow

* Don't use `nd_range` parallel for unless you absolutely have to, as it is difficult to map efficiently to CPUs. 
* If you don't need barriers or local memory, use `parallel_for` with `range` argument.
* If you need local memory or barriers, scoped parallelism or hierarchical parallelism models may perform better on CPU than `parallel_for` kernels using `nd_range` argument and should be preferred. Especially scoped parallelism also works well on GPUs.
* If you *have* to use `nd_range parallel_for` with barriers on CPU, the `omp.accelerated`  or `generic` compilation flow will most likely provide substantially better performance than the `omp.library-only` compilation target. See the [documentation on compilation flows](doc/compilation.md) for details.


## Strong-scaling/latency-bound problems

* Eager submission can be forced by setting the environment variable `ACPP_RT_MAX_CACHED_NODES=0`. By default AdaptiveCpp performs batched submission.
* The alternative instant task submission mode can be used, which can substantially lower task launch latencies. Define the macro `HIPSYCL_ALLOW_INSTANT_SUBMISSION=1` before including `sycl.hpp` to enable it. Instant submission is possible with operations that do not use buffers (USM only), have no dependencies on non-instant tasks, do not use SYCL 2020 reductions and use in-order queues. In the stdpar model, instant submission is active by default.
* SYCL 2020 `in_order` queues bypass certain scheduling layers and may thus display lower submission latency.
* The USM pointer-based memory management model typically has less overheads and lower latency compared to SYCL's traditional buffer-accessor model.
* Consider using the `HIPSYCL_EXT_COARSE_GRAINED_EVENTS` [(extension documentation)](extensions.md) extension if you rarely use events returned from the `queue`. This extension allows the runtime to elide backend event creation.
* Stdpar kernels typically have lower submission latency compared to SYCL kernels.

## Stdpar

* For performance in the C++ parallelism model specifically, see also [here](doc/stdpar.md).
