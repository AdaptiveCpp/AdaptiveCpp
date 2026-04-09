# Deploying AdaptiveCpp-generated binaries

AdaptiveCpp provides mechanisms to aid in deploying AdaptiveCpp-compiled binaries to end users. The following discussion focuses on applications compiled by the generic SSCP compiler (`--acpp-targets=generic`), but other compilation flows might work as well.

## Deployment infrastructure

Developers can use `acpp --acpp-deploy` to populate a directory with runtime dependencies of AdaptiveCpp applications.

The contents of that directory can then be distributed along with the application. Please ensure that the application uses this directory to resolve library dependencies, e.g. by setting `LD_LIBRARY_PATH`.

### Deployment components

`--acpp-deploy` distinguishes separate deployment targets that can be deployed either individually or all at once:

* `core`: Core infrastructure, CPU backend and dependencies
* `cuda`: CUDA backend and dependencies
* `hip`: HIP backend and dependencies
* `ocl`: OpenCL backend and dependencies
* `all`: All of the above.

**The `core` component always needs to be available for an AdaptiveCpp-compiled application to function**. The other components are optional.

Because applications compiled with AdaptiveCpp's generic SSCP compiler are decoupled from hardware targets and specific backends, the optional deployment components can also be installed by end users at any later point in time. For example, if users change their hardware setup, the same application will work on the hardware once users install the required component.

In order to minimize the size of the deployment package for end users, it might be a good idea for application vendors to provide the optional components only as needed (e.g. as optional selections in an installation wizard).

**Note on the `ocl` component:** AdaptiveCpp only handles the deployment of the OpenCL ICD loader. End users are responsible for installing an OpenCL driver for their hardware, as in the established deployment model for OpenCL applications.

### Invoking `--acpp-deploy`

A deployment package can be generated using `acpp --acpp-deploy=<component>:<path>`. For example, 

```
acpp --acpp-deploy=all:/my/deployment/path
```

will deploy all components to `/my/deployment/path`. You should then be able to run your binary e.g. using `LD_LIBRARY_PATH=/my/deployment/path:$LD_LIBRARY_PATH ./myapplication`.

**Note that for a successful deployment of a component, AdaptiveCpp must have been built with the respective backend enabled!** We can only deploy a component if we have it :-)

## Limitations and handling indirect transitive dependencies

In the deployment package, depending on which component was requested for deployment, AdaptiveCpp includes:

* AdaptiveCpp runtime libraries
* AdaptiveCpp runtime backends
* Infrastructure for the JIT compiler, including bitcode libraries and needed LLVM components
* Backend-specific dependencies (e.g. needed CUDA runtime libraries, necessary components from ROCm etc)

However, for LLVM and backend dependencies, AdaptiveCpp cannot know in detail how exactly those were built and which transitive dependencies these may have in all cases.
The AdaptiveCpp deployment mechanism includes needed dependencies for common setups, but this may be insufficient depending on how self-contained you would like your package to be.

After a successful deployment, `acpp` will recommend that you run a command to check for dependencies outside of your deployment tree. This command will look similar to

```sh
LD_LIBRARY_PATH=/some/deployment/path:$LD_LIBRARY_PATH ldd `find /some/deployment/path -type f | grep -v .bc` | grep -v /some/deployment/path | awk '{print $3;}' | sort | uniq
```

Libraries listed by this command are additional transitive dependencies of libraries in the deployment tree that you may want to consider including as well.

However, you should **not** include the following libraries:
* Core system libraries: `libc`, `linux-vdso`
* `libcuda.so`, as it is provided by the NVIDIA graphics driver and needs to match it
* `libdrm*`, as it too is part of the graphics driver stack.

When including additional dependencies, even if they were detected as transitive dependencies of specific backends, we recommend including them in the `core` package *if* you decide to ship separate deployment packages. The reason is that this can avoid file conflicts if several components pull in the same dependency.

## Decreasing the size of the deployment package

The size of the deployment package is typically strongly dominated by the size of `libLLVM.so`, which on its own can reach around 150MB in size.

The `hip` deployment component specifically is expected to pull in a library from ROCm (`libamd_comgr`) which may be statically linked against ROCm's LLVM and might thus again be around another 150MB in size.

A deployment package for all components/backends is therefore expected to be at a little over 300MB in size. Attempts to optimize this should focus on libLLVM, and if the `hip` component is included, on comgr.

Two ways to improve on this are:
1. LLVM, as shipped by many distributions, typically includes support for all available compiler backends for cross-compilation use cases. However, AdaptiveCpp only needs the backend for the host CPU (e.g. X86), NVPTX, AMDGPU and potentially spir64. So, building a custom LLVM with only these backends enabled might reduce binary size.
2. A very quick, convenient and highly effective solution is to use a binary packer like e.g. [upx](https://upx.github.io/), which can compress libraries transparently such that they automatically decompress in memory when they are needed. upx-compressed libLLVM often achieves compression rates of around 40% and can thus almost cut the size of the deployment package in half!


## Forward compatibility and updating the deployment package

When deploying applications to end users, it is typically desired that the application should continue to work if the user upgrades their hardware, potentially even to hardware that was not available yet when the application was originally distributed.

This notion of forward-compatibility is currently supported to the following extent:

* OpenMP CPU backend: Yes, however performance may not be ideal if AdaptiveCpp's LLVM is too unfamiliar with the CPU architecture (same as with regular C++ applications when the compiler is too old).
* CUDA backend: Yes, however this is only lightly tested at the moment.
* OpenCL backend: Yes, without limitations.
* HIP backend: Due to lack of forward compatibility in ROCm, AdaptiveCpp can only target those AMD GPUs that are supported by the ROCm version it was built against.

In many cases, it is possible to simply generate a new deployment package with newer, updated backend dependencies and use the new package without recompiling the application.
In order to achieve this, rebuild the same AdaptiveCpp version against the updated dependencies (e.g. ROCm) and rerun deployment. The new package may the be distributed to users.

### Upgrading the HIP deployment package
It is possible to update the HIP deployment package with support for newer hardware, and provide the user with the updated package. For example, you could rebuild AdaptiveCpp against a newer ROCm version, rerun deployment for the HIP component and give users the updated package.
The same, unmodified application should then be able to run on the new hardware.

### Upgrading the CUDA and OpenCL package

You can also upgrade the CUDA and OpenCL deployment packages, however this should rarely be necessary since those platforms already provide good forward compatibility.

### Upgrading the core package and LLVM

The same is possible with LLVM updates, *if* your AdaptiveCpp version already supports the newer LLVM version that you want to target.
**It is however not in general possible to update the AdaptiveCpp version without recompiling the application, because the AdaptiveCpp runtime is not guaranteed to have a stable ABI!**
So, if the version of AdaptiveCpp that you have built the application binary with does not yet support the LLVM that you want, then you will have to update AdaptiveCpp itself, recompile the application, and provide users with the new binary.

### Clearing the JIT cache

After an update of the deployment package, it might be a good idea to instruct users to clear the AdaptiveCpp JIT cache (or have some install wizard do this) to avoid outdated kernels being passed to drivers.

## CUDA redistribution

Note that the deployment mechanism pulls in components from backends which, in the case of CUDA, are not under an open source license. However, all CUDA components utilized by AdaptiveCpp and deployed as part of the deployment mechanism are explicitly cleared for redistribution in the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html#attachment-a).
Nevertheless, you may still want to be aware that portions of the software distributed by you may be covered by the CUDA EULA terms.

## Vector math library redistribution

The deployment mechanism may include and redistribute third-party libraries under the following licenses:

- Intel Short Vector Math Library (`svml.so` & `intlc.so`)
  Provided under the [Intel End User License Agreement (EULA)](https://www.intel.com/content/www/us/en/content-details/777700/intel-end-user-license-agreement-for-developer-tools.html)

- Arm Performance Libraries Math Library (`amath.so`)
  Provided under the [Arm Performance Libraries End User License Agreement](https://developer.arm.com/documentation/109611/1-0/End-User-License-Agreement--EULA-?lang=en)

- SLEEF Vector Math Library (`sleef.so`)
  Provided under the [Boost Software License, Version 1.0](https://www.boost.org/LICENSE_1_0.txt)
