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

### Ubuntu/Debian Example (LLVM 20)

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
sudo apt install -y libclang-20-dev clang-tools-20 libomp-20-dev llvm-20-dev lld-20
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




