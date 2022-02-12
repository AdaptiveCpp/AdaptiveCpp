# hipSYCL compilation model


hipSYCL relies on the fact that many existing programming models as well as and SYCL are single-source programming models based on C++. This means that it is possible to extend existing toolchains, such as CUDA and HIP toolchains, to also support SYCL code. hipSYCL does that by using clang's CUDA and HIP toolchains with a custom clang plugin. Additionally, hipSYCL can utilize the clang SYCL frontend to generate SPIR-V code. 
hipSYCL contains mechanisms to embed and aggregate  compilation results from multiple toolchains into a single binary, allowing it to effectively combine multiple toolchains into one. This is illustrated here:

![syclcc design](/doc/img/syclcc.png)

hipSYCL distinguishes two modes of compilation:
* *integrated multipass*, where host and device compilation passes are handled by clang's CUDA and HIP drivers. This mode allows for the most interoperability with backends because backend-specific language extensions are also available in the host pass. For example, kernels can be launched using the `<<<>>>` syntax. However, limitations in clang's compilation drivers also affect hipSYCL in this mode. In particular, CUDA and HIP cannot be targeted simultaneously because host code cannot support language extensions from *both* at the same time.
* *explicit multipass*, where host and device compilation passes are handled by hipSYCL's `syclcc` compiler driver. Here, `syclcc` invokes backend-specific device passes, which then result in a compiled kernel image for the target device. `syclcc` then embeds the compiled kernel image into the host binary. At runtime, the kernel is extracted from the image and invoked using runtime functions instead of language extensions. As a consequence, explicit multipass compilation for one backend can be combined with arbitrary other backends simultaneously, allowing hipSYCL to target multiple device backends at the same time. Note that in explicit multipass, different guarantees might be made regarding the availability of backend-specific language extensions in the host pass compared to integrated multipass. See the section on language extension guarantees below for more details.

Not all backends support all modes:


| Backend      | Integrated multipass? | Explicit multipass? | Comment                                  |
|--------------|-----------------------|---------------------|------------------------------------------|
| OpenMP       | N/A                   | N/A                 | Does not require multipass compilation   |
| CUDA         | Yes                   | Yes                 |                                          |
| CUDA (nvc++) | Yes                   | No                  | No support for CPU nd-range acceleration |
| HIP          | Yes                   | No                  |                                          |
| SPIR-V       | No                    | Yes                 |                                          |

**Note:** Explicit multipass requires building hipSYCL against a clang that supports `__builtin_unique_stable_name()` (available in clang 11), or clang 13 or newer as described in the [installation documentation](installing.md).

## Language extension guarantees

hipSYCL allows using backend-specific language extensions (e.g. CUDA/HIP C++). The precise guarantees about the availability of these extensions are as follows:

* If a backend runs on a compiler that provides a unified, single compilation pass for both host and device, backend-specific language extensions are always available. Currently this only affects the CUDA-nvc++ backend.
* If the compiler relies on separate compilation passes for host and device:
  * In device compilation passes, backend-specific language extensions are always available.
  * In host compilation passes, the following applies:
    * If the backend runs in integrated multipass mode, backend-specific language extensions are available.
    * If the backend runs in explicit multipass mode:
      * For SPIR-V, language extensions are always available
      * For CUDA and HIP: Language extensions from *one* of them are available in the host pass.
        * If one of them runs in integrated multipass and one in explicit multipass, language extensions from the one in integrated multipass are available
        * If both are in explicit multipass, `syclcc` will currently automatically pick one that will have language extensions enabled in the host pass.




## Compiler support to accelerate nd_range parallel_for on CPUs
The nd_range parallel_for is not efficiently implementable without explicit compiler support.
hipSYCL provides multiple options to circumvent this, though.
1. The recommended and library-only solution is writing kernels using a performance portable
paradigm as hipSYCL's [scoped parallelism](scoped-parallelism.md) extension.
2. If that is not an option (e.g. due to preexisting code), hipSYCL provides a compiler extension that allows efficient 
execution of the nd_range paradigm at the cost of forcing the host compiler to Clang.
3. Without the Clang plugin, a Boost.Fiber implementation of the nd_range paradigm will be used.
This will come with a significant performance hit, though, as soon as a barrier is used inside the kernel.

For the compiler extension variant, the hipSYCL Clang plugin implements a set of passes to perform deep loop fission
on nd_range parallel_for kernels that contain barriers. The continuation-based synchronization
approach is employed to achieve good performance and functional correctness (_Karrenberg, Ralf, and Sebastian Hack. "Improving performance of OpenCL on CPUs." International Conference on Compiler Construction. Springer, Berlin, Heidelberg, 2012. [https://link.springer.com/content/pdf/10.1007/978-3-642-28652-0_1.pdf](https://link.springer.com/content/pdf/10.1007/978-3-642-28652-0_1.pdf)_).
A deep dive into how the implementation works and why this approach was chosen
can be found in Joachim Meyer's [master thesis](https://joameyer.de/hipsycl/Thesis_JoachimMeyer.pdf).

As Clang plugins are only officially supported on Linux, this functionality is currently
supported by hipSYCL on Linux only as well.
By default, this compiler support is used on Linux.
It can be explicitly disabled, by setting `-DWITH_ACCELERATED_CPU=OFF` at hipSYCL build time.
If hipSYCL was built with the compiler support enabled, it can be explicitly en-/disabled
by adding `--hipsycl-use-accelerated-cpu`/`--hipsycl-use-accelerated-cpu=false` to the 
syclcc command line. When using CMake, set `-DHIPSYCL_USE_ACCELERATED_CPU=` to override
the default behavior.
