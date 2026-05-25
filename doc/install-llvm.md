# LLVM Dependency Guide

AdaptiveCpp depends on LLVM/Clang for its core compilation flows. It uses a Clang plugin to provide SYCL support and leverages LLVM's JIT infrastructure for the generic SSCP flow.

## Compatibility Matrix

| LLVM Version | Profile | Supported Flows |
| :--- | :--- | :--- |
| **LLVM 15 - 20** | `full` | **Recommended**. Support for all backends and SSCP. |
| **LLVM < 15** | `minimal` | Limited support for older legacy flows only. |
| **Dev/Unreleased** | - | **Not Supported**. |

> [!WARNING]
> **AdaptiveCpp does not support unreleased or development versions of LLVM.** Always use official release branches (e.g., `release/20.x`).

---

## Recommended Installation (Linux)

For most users, the LLVM version provided by your distribution (if >= 15) or [apt.llvm.org](http://apt.llvm.org) is the best choice.

For example, the required steps to install clang 21 on an Ubuntu system are:
```
wget https://apt.llvm.org/llvm.sh #Convenience script that sets up the repositories
chmod +x llvm.sh
./llvm.sh 21 #Set up repositories for clang 21
apt install -y libclang-21-dev clang-tools-21 libomp-21-dev llvm-21-dev lld-21
```

#### Only if you wish to compile LLVM from source (not recommended)

It is generally not necessary to compile LLVM by yourself. However, if you wish to do this, during LLVM cmake make sure to:

- Disable assertions as AdaptiveCpp can potentially trigger some (false positive) debug assertions in some LLVM versions: `-DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_DUMP=OFF` 
- Generate `libLLVM.so`: `-DLLVM_BUILD_LLVM_DYLIB=ON` (only required if the SSCP compilation flow is enabled when building AdaptiveCpp, which is true by default for supported versions of LLVM)
- Enable the correct backends for your hardware: `nvptx` for NVIDIA GPUs and `amdgpu` for AMD GPUs.

An example build of LLVM 21 from source might look like this:

```
git clone https://github.com/llvm/llvm-project -b release/21.x
cd llvm-project
mkdir -p build
cd build

INSTALL_PREFIX=/path/to/desired/llvm/installation/directory

cmake -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DLLVM_ENABLE_PROJECTS="clang;lld;openmp" \
      -DLLVM_ENABLE_RUNTIMES=compiler-rt \
      -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86" \
      -DLLVM_INCLUDE_BENCHMARKS=0 \
      -DLLVM_INCLUDE_EXAMPLES=0 \
      -DLLVM_INCLUDE_TESTS=0 \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_INSTALL_RPATH=$INSTALL_PREFIX/lib \
      -DLLVM_ENABLE_OCAMLDOC=OFF \
      -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
      -DLLVM_BUILD_LLVM_DYLIB=ON \
      -DLLVM_ENABLE_DUMP=OFF \
      ../llvm

make install
```

---

## Configuring CMake for LLVM

AdaptiveCpp attempts to detect LLVM automatically. If it fails or you have multiple versions, use these variables:

| Variable | Description |
| :--- | :--- |
| `-DLLVM_DIR` | Path to the LLVM CMake directory (e.g., `/usr/lib/llvm-20/lib/cmake/llvm`). |
| `-DLLVM_ROOT` | Root path of the LLVM installation. |
| `-DCLANG_EXECUTABLE_PATH` | Path to the `clang++` binary. |
| `-DCLANG_INCLUDE_PATH` | Path to internal Clang headers. |

### Verification
After running CMake, check the output to ensure the correct version was selected:
```text
-- Building AdaptiveCpp against LLVM configured from /usr/lib/llvm-20/cmake
-- Selecting clang: /usr/lib/llvm-20/bin/clang++
```

---

## Troubleshooting

### Missing C++ Standard Library Headers
Clang does not ship with its own C++ standard library (`libstdc++` or `libc++`). It uses the one found on the system.

- **Missing packages**: Ensure `libstdc++-dev` or `g++` is installed.
- **Ambiguous selection**: If Clang picks the wrong GCC toolchain, use the `--gcc-toolchain` flag.

> [!TIP]
> Use `clang++ -v /dev/null` to see which GCC installation Clang is currently selecting.

### Compiling LLVM from Source (Not Recommended)
If you must build LLVM yourself, ensure the following flags are used:
- `-DLLVM_BUILD_LLVM_DYLIB=ON` (Required for SSCP)
- `-DLLVM_ENABLE_PROJECTS="clang;lld;openmp"`
- `-DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86"` (Include your target GPUs)
- `-DCMAKE_BUILD_TYPE=Release`
- `-DLLVM_ENABLE_ASSERTIONS=OFF`




