# hipSYCL installation notes for ROCm

In general, hipSYCL on ROCm requires a clang installation that satisfies the following conditions:
* clang must **not** be compiled with the `_DEBUG` macro enabled (otherwise, hipSYCL triggers incorrect `assert()` statements inside clang due to the, from clang's perpective, unusual way in which it employs the HIP/CUDA toolchain). Regular clang release builds as they can be found in distributions usually meet this requirement.
* clang versions older than 8 are not supported because HIP support was introduced in 8.
* clang must have been built with the AMDGPU backend enabled (which is usually the case)
* lld is also required.
* ROCm versions older than 2.6 cannot work, but recent hipSYCL distributions may or may not work with such old ROCm stacks. It is recommended to use a recent ROCm version.

Additionally, please take note of the following:
* **regular LLVM/clang distributions:** hipSYCL can run with regular clang/LLVM distributions, as long as you have matching ROCm device and runtime libraries installed. This is usually **the easiest way and therefore recommended.**
* **clang distributions from AMD:**
  * In principle, clang forks from AMD like `aomp` releases can be used as well, however at least the aomp 0.7 binary packages are compiled with the `_DEBUG` macro enabled, which is unsupported as described above. Additionally, the way that aomp currently advertises its version causes cmake's version identification to discard it as incompatible.

Once you have a suitable clang installed, make sure to compile hipSYCL against this clang. 

The following cmake variables may be relevant:
* Use `ROCM_PATH` to point hipSYCL to ROCm, if you haven't installed it in /opt/rocm.



