# hipSYCL installation notes for ROCm

In general, hipSYCL on ROCm requires a clang installation that satisfies the following conditions:
* clang must **not** be compiled with the `_DEBUG` macro enabled (otherwise, hipSYCL triggers incorrect `assert()` statements inside clang due to the, from clang's perpective, unusual way in which it employs the HIP/CUDA toolchain). Regular clang release builds as they can be found in distributions usually meet this requirement.
* clang versions older than 8 are not supported
* clang must have been built with the AMDGPU backend enabled (which is usually the case)
* lld is also required.

Additionally, please take note of the following:
* **regular LLVM/clang distributions:** hipSYCL can run with regular clang/LLVM distributions, as long as you have matching ROCm device and runtime libraries installed. This is usually **the easiest way and therefore recommended.** The following combinations of mainline clang and ROCm are known to work:
  * clang 10 with ROCm 2.6
  * clang 9 with ROCm 2.6
  * HIP support was introduced in clang 8, so anything older than clang 8 cannot work.
  If you use a debian-based Linux distribution (e.g. Debian/Ubuntu), you can get binary packages for llvm/clang from [apt.llvm.org](http://apt.llvm.org). Note that currently the clang 9 packages seem to be packaged incorrectly, preventing correct detection of the installation by hipSYCL's cmake - you will have to use the nightly clang 10 snapshots.
* **clang distributions from AMD:**
  * In principle, clang forks from AMD like `aomp` releases can be used as well, however at least the aomp 0.7 binary packages are compiled with the `_DEBUG` macro enabled, which is unsupported as described above. Additionally, the way that aomp currently advertises its version causes cmake's version identification to discard it as incompatible (we specifically ask for clang/LLVM 8,9 or 10).

Once you have a suitable clang installed, make sure to compile hipSYCL against this clang. 

The following cmake variables may be relevant:
* Use `ROCM_PATH` to point hipSYCL to ROCm, if you haven't installed it in /opt/rocm.



