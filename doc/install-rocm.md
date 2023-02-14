# Open SYCL installation instructions for ROCm

Please install ROCm 4.0 or later as described in the ROCm readme. Make sure to also install HIP (runtime libraries and headers).

*Note: Newer ROCm versions may require building Open SYCL against newer clang versios as well. For example, ROCm 4.5 requires clang 13+.*

*Note: Instead of building Open SYCL against a regular clang/LLVM, it is also possible to build Open SYCL against the clang/LLVM that ships with ROCm. This can be interesting if other available clang/LLVM installations are not new enough to work with the ROCm installation.* 
* **Such configurations typically work, but are generally less tested.**
* Also note that the LLVM distributions shipping with ROCm are not official LLVM releases, and depending on when the upstream development was last merged, may have slightly diverging functionality. There are multiple known cases where this causes problems: 
  * The clang 13 from ROCm 4.5 lacks functionality that is present in official clang 13 releases and that Open SYCL's clang 13 code paths need. In that case you will need to set `-DHIPSYCL_NO_DEVICE_MANGLER=ON` when compiling Open SYCL. This will however break [explicit multipass](compilation.md) support.
  * Similarly, the clang 14 from ROCm 5.0 lacks functionality that is present in official clang 14 releases. You can work around those issues by setting `-DWITH_ACCELERATED_CPU=OFF -DWITH_SSCP_COMPILER=OFF` at the expense of reduced kernel performance on CPUs and lack of [SSCP](compilation.md) support.

*Note: Open SYCL is by default configured to utilize the ROCm compilation flags that apply for recent clang and ROCm versions. If you are using an older clang (<= 10) or ROCm < 4, you might have to adjust `-DROCM_CXX_FLAGS` (not recommended!).*

CMake variables:
* `-DROCM_PATH=/path/to/rocm` (default: /opt/rocm)
* `-DWITH_ROCM_BACKEND=ON` if Open SYCL does not automatically enable the ROCm backend 
* `-DHIPSYCL_NO_DEVICE_MANGLER=OFF/ON` *if and only if* you build against ROCm's clang and hit the issue that it lacks functionality that regular clang 13 provides, and you cannot build Open SYCL otherwise. This *will* break [explicit multipass](compilation.md) support, i.e. you will not be able to compile for multiple device backends simultaneously.

