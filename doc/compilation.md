# hipSYCL compilation model


hipSYCL relies on the fact that both HIP/CUDA and SYCL are single-source programming models based on C++. This means that it is possible to extend existing toolchains CUDA and HIP toolchains to also support SYCL code. hipSYCL does that by using clang's CUDA and HIP toolchains with a custom clang plugin.

hipSYCL distinguishes two modes of compilation:
* *integrated multipass*, where host and device compilation passes are handled by clang's CUDA and HIP drivers. This mode allows for the most interoperability with backends because backend-specific language extensions are also available in the host pass. For example, kernels can be launched using the `<<<>>>` syntax. However, limitations in clang's compilation drivers also affect hipSYCL in this mode. In particular, CUDA and HIP cannot be targeted simultaneously because host code cannot support language extensions from *both* at the same time.
* *explicit multipass*, where host and device compilation passes are handled by hipSYCL's `syclcc` compiler driver. Here, `syclcc` invokes backend-specific device passes, which then result in a compiled kernel image for the target device. `syclcc` then embeds the compiled kernel image into the host binary. At runtime, the kernel is extracted from the image and invoked using runtime functions instead of language extensions. Therefore, in the host pass only C++ support is required. As a consequence, explicit multipass compilation for one backend can be combined with arbitrary other backends simultaneously, allowing hipSYCL to target multiple device backends at the same time.

Not all backends support all modes:


| Backend | Integrated multipass? | Explicit multipass? | Comment |
|------------------|-------------------|------------------|------------------|
| OpenMP | N/A | N/A | Does not require multipass compilation |
| CUDA   | Yes | Yes |  |
| HIP   | Yes | No |  |
| SPIR-V  | No | Yes | |

**Note:** Explicit multipass requires building hipSYCL against a clang that supports `__builtin_unique_stable_name()` as described in the [installation documentation](installing.md).

