# Advanced Installation Methods

This guide covers advanced and experimental installation methods for AdaptiveCpp, such as building as part of LLVM or using multi-stage bootstrap builds.

## Building an LLVM toolchain with AdaptiveCpp linked in

> [!NOTE]
> This method is currently considered experimental, but is the recommended approach for Windows support.

Building AdaptiveCpp as part of LLVM makes it easy to ship a full installation with all dependencies. It also enables systems where LLVM plugins are limited (like Windows) to use most compiler features.

When building this way, AdaptiveCpp's compiler components are linked into the LLVM tools (`clang`, `opt`, etc.), so separate LLVM plugins are not required.

### Instructions

1.  Select a released LLVM version (e.g., LLVM 18).
2.  Clone both LLVM and AdaptiveCpp into the same hierarchy.
3.  Use `ninja` for the build.

```bash
export LLVM_VERSION=18 # set me!
export LLVM_PARALLEL_LINK_JOBS=8 # set me (allow ~4GB RAM per link job)
export ACPP_INSTALL_PREFIX=`pwd`/../../install # set me
export USE_CCACHE=ON # optional, for faster rebuilds

git clone https://github.com/llvm/llvm-project --single-branch -b release/${LLVM_VERSION}.x llvm
cd llvm
git clone https://github.com/AdaptiveCpp/AdaptiveCpp AdaptiveCpp
mkdir -p build && cd build

cmake ../llvm -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_PREFIX} \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_PROJECTS="clang;openmp;lld" \
  -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS} \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_CCACHE_BUILD=${USE_CCACHE} \
  -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp \
  -DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=`pwd`/../AdaptiveCpp \
  -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON

ninja install
```

### Caveats
- **Windows**: Requires LLVM 18 or above.
- **Combined Targets**: Using `omp` + `hip` together requires LLVM 17+.
- **Stability**: This is experimental; please test your specific use case.

---

## 2-Stage Bootstrap Build (macOS)

On systems where the default compiler has ABI incompatibilities with separately distributed compilers (e.g., macOS with Homebrew LLVM), a 2-stage compilation process (bootstrapping) is recommended.

This ensures that LLVM and AdaptiveCpp are built using a consistent toolchain, which is often necessary for the generic JIT compiler to work correctly on Apple Silicon.

### Prerequisites

You will need `ninja`, `cmake`, and `boost`. On macOS, use Homebrew:

```bash
# Note: CMake 4.0 may break LLVM's compiler-rt build; use 3.31 if possible
brew install boost ninja cmake
```

### Instructions

```bash
export LLVM_VERSION=18
export LLVM_PARALLEL_LINK_JOBS=4
export ACPP_INSTALL_PREFIX=`pwd`/../../install
export TARGETS_TO_BUILD="AArch64" # Default for Apple Silicon (M1/M2/M3)

git clone https://github.com/llvm/llvm-project --single-branch -b release/${LLVM_VERSION}.x llvm
cd llvm
git clone https://github.com/AdaptiveCpp/AdaptiveCpp AdaptiveCpp
mkdir -p build && cd build

cmake ../llvm -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_PREFIX}  \
        -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
        -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS} \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DCLANG_ENABLE_BOOTSTRAP=ON \
        -DCLANG_BOOTSTRAP_PASSTHROUGH="CMAKE_INSTALL_PREFIX;LLVM_TARGETS_TO_BUILD;LLVM_PARALLEL_LINK_JOBS;LLVM_BUILD_LLVM_DYLIB;LLVM_LINK_LLVM_DYLIB" \
        -DBOOTSTRAP_LLVM_ENABLE_PROJECTS="clang;lld;openmp" \
        -DBOOTSTRAP_LLVM_EXTERNAL_PROJECTS=AdaptiveCpp \
        -DBOOTSTRAP_LLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=`pwd`/../AdaptiveCpp \
        -DBOOTSTRAP_LLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON \
        -DBOOTSTRAP_WITH_OPENCL_BACKEND=OFF 

ninja stage2-install
```

> [!TIP]
> To pass CMake flags to AdaptiveCpp in a 2-stage build, prefix them with `BOOTSTRAP_`. For example, use `-DBOOTSTRAP_WITH_CUDA_BACKEND=ON`.
