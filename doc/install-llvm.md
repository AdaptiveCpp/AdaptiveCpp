# LLVM dependency installation instructions

Certain backends/compilation flows require LLVM. This is because AdaptiveCpp needs to build a clang plugin which is then loaded into clang to provide the required compiler support for SYCL.

Generally, we recommend the latest officially released clang/LLVM versions, but older versions might also work depending on the compilation flow (see the table from the main installation instructions).

Usually, the clang/LLVM versions provided in Linux distribution repositories are sufficient, if they are recent enough. 
*In this case, AdaptiveCpp might automatically detect and configure your LLVM installation without additional cmake arguments required.* **We therefore recommend to make your life easy: Check your distribution's LLVM version against the AdaptiveCpp requirements and if they match, use it**.

If you are using Ubuntu or Debian, we can also recommend the package repositories at `http://apt.llvm.org` if you wish to obtain a newer LLVM.

Install
* clang (including development headers)
* LLVM (including development headers)
* libomp (including development headers)
* lld (only for the ROCm backend)

For example, the required steps to install clang 13 on an Ubuntu system are:
```
wget https://apt.llvm.org/llvm.sh #Convenience script that sets up the repositories
chmod +x llvm.sh
./llvm.sh 13 #Set up repositories for clang 13
apt install -y libclang-13-dev clang-tools-13 libomp-13-dev llvm-13-dev lld-13
```

#### Only if you wish to compile LLVM from source (not recommended)

It is generally not necessary to compile LLVM by yourself. However, if you wish to do this, during LLVM cmake make sure to:

- Disable assertions as AdaptiveCpp can potentially trigger some (false positive) debug assertions in some LLVM versions: `-DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_DUMP=OFF` 
- Generate `libLLVM.so`: `-DLLVM_BUILD_LLVM_DYLIB=ON` (only required if the SSCP compilation flow is enabled when building AdaptiveCpp, which is true by default for supported versions of LLVM)
- Enable the correct backends for your hardware: `nvptx` for NVIDIA GPUs and `amdgpu` for AMD GPUs.

## Pointing AdaptiveCpp to the right LLVM

When invoking cmake, the AdaptiveCpp build infrastructure will attempt to find LLVM automatically (see below for how to invoke cmake).

If AdaptiveCpp does not automatically configure the build for the desired clang/LLVM installation, the following cmake variables can be used to point AdaptiveCpp to the right one:
* `-DLLVM_DIR=/path/to/llvm/cmake` must be pointed to your LLVM installation, specifically, the **subdirectory containing the LLVM cmake files**. Note that different LLVM installations may have the LLVM cmake files in different subdirectories that don't necessarily end with `cmake` (e.g. it might also be `/path/to/llvm/lib/cmake/llvm`). Alternatively, you can try `-DLLVM_ROOT` which might be more forgiving.

Verify from the cmake that the selected `clang++` and include headers match the LLVM that you have requested. Example output:
```
...
-- Building AdaptiveCpp against LLVM configured from /usr/lib/llvm-13/cmake/
-- Selecting clang: /usr/bin/clang++-13
-- Using clang include directory: /usr/include/clang/13.0.1/include/..
...
```

If AdaptiveCpp does not select the right clang++ or include directories, use the following variables to set them manually:


* `-DCLANG_EXECUTABLE_PATH=/path/to/clang++` must be pointed to the `clang++` executable from this LLVM installation.
* `-DCLANG_INCLUDE_PATH=/path/to/clang-includes` must be pointed to the clang internal header directory. Typically, this is something like `$LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include`. Newer ROCm versions will require the parent directory instead, i.e. `$LLVM_INSTALL_PREFIX/include/clang/<llvm-version>`. This is only important for the ROCm backend.
