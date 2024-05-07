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

For example, the required steps to install clang 16 on an Ubuntu system are:
```
wget https://apt.llvm.org/llvm.sh #Convenience script that sets up the repositories
chmod +x llvm.sh
./llvm.sh 16 #Set up repositories for clang 16
apt install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev lld-16
```

#### Only if you wish to compile LLVM from source (not recommended)

It is generally not necessary to compile LLVM by yourself. However, if you wish to do this, during LLVM cmake make sure to:

- Disable assertions as AdaptiveCpp can potentially trigger some (false positive) debug assertions in some LLVM versions: `-DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_DUMP=OFF` 
- Generate `libLLVM.so`: `-DLLVM_BUILD_LLVM_DYLIB=ON` (only required if the SSCP compilation flow is enabled when building AdaptiveCpp, which is true by default for supported versions of LLVM)
- Enable the correct backends for your hardware: `nvptx` for NVIDIA GPUs and `amdgpu` for AMD GPUs.

An example build of LLVM 15 from source might look like this:

```
git clone https://github.com/llvm/llvm-project -b release/15.x
cd llvm-project
mkdir -p build
cd build

INSTALL_PREFIX=/path/to/desired/llvm/installation/directory

cmake -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;openmp" \
      -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86" \
      -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
      -DLLVM_INCLUDE_BENCHMARKS=0 \
      -DLLVM_INCLUDE_EXAMPLES=0 \
      -DLLVM_INCLUDE_TESTS=0 \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_INSTALL_RPATH=$INSTALL_PREFIX/lib \
      -DLLVM_ENABLE_OCAMLDOC=OFF \
      -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
      -DLLVM_BUILD_LLVM_DYLIB=ON \
      -DLLVM_ENABLE_DUMP=OFF  ../llvm
make install
```

## Pointing AdaptiveCpp to the right LLVM

When invoking cmake, the AdaptiveCpp build infrastructure will attempt to find LLVM automatically (see below for how to invoke cmake).

If AdaptiveCpp does not automatically configure the build for the desired clang/LLVM installation, the following cmake variables can be used to point AdaptiveCpp to the right one:
* `-DLLVM_DIR=/path/to/llvm/cmake` must be pointed to your LLVM installation, specifically, the **subdirectory containing the LLVM cmake files**. Note that different LLVM installations may have the LLVM cmake files in different subdirectories that don't necessarily end with `cmake` (e.g. it might also be `/path/to/llvm/lib/cmake/llvm`). Alternatively, you can try `-DLLVM_ROOT` which might be more forgiving.

Verify from the cmake that the selected `clang++` and include headers match the LLVM that you have requested. Example output:
```
...
-- Building AdaptiveCpp against LLVM configured from /usr/lib/llvm-16/cmake/
-- Selecting clang: /usr/bin/clang++-16
-- Using clang include directory: /usr/include/clang/16.0.1/include/..
...
```

If AdaptiveCpp does not select the right clang++ or include directories, use the following variables to set them manually:


* `-DCLANG_EXECUTABLE_PATH=/path/to/clang++` must be pointed to the `clang++` executable from this LLVM installation.
* `-DCLANG_INCLUDE_PATH=/path/to/clang-includes` must be pointed to the clang internal header directory. Typically, this is something like `$LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include`. Newer ROCm versions will require the parent directory instead, i.e. `$LLVM_INSTALL_PREFIX/include/clang/<llvm-version>`. This is only important for the ROCm backend.

## If clang/LLVM does not find C++ standard library headers

AdaptiveCpp relies on a working clang installation. clang in turn does not come with its own C++ standard library, but instead assumes that it can use the one from the system.

### Missing packages

If it does not find C++ standard library headers, it might be because you are missing a `libstdc++-dev` package or similar.

If you have multiple gcc versions installed on your system, it is also possible that clang wants a specific libstdc++ version to match a gcc version that it has selected. You can use `clang++ -v /dev/null` to find out which gcc and libstdc++ version clang wants to use.
*Note: This command is expected to fail with a linker error. It should print something like "Selected GCC installation: ..." which is the information you are looking for.*

### GCC toolchain/C++ standard library is in a non-standard location

If the C++ standard library is in a non-standard location (this might typically be the case on a cluster), it is also possible that clang does not find it.

In that case, clang can be informed about the location of the GCC toolchain using the `--gcc-toolchain=...` flag.
*Note:* This flag is very picky about the directory!. It needs to be provided with the parent directory that contains the `lib/gcc/<arch>/<version>` subdirectories. On a Cray system, this might e.g. be `--gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos`. If an incorrect path is provided, clang may silently ignore the argument. Use `clang++ -v /dev/null` as described above to check that it has accepted the GCC installation.

While in theory AdaptiveCpp can be made to use that flag when compiling, a simpler solution to make that change permanent is to use clang configuration files ([details in the clang documentation](https://clang.llvm.org/docs/UsersManual.html#id25)):
1. In the directory where `clang++` lives (e.g. some `bin` directory), create a new file `<name>.cfg`, replacing `<name>` with something of your choice.
2. Put any flags that you want clang to use by default into this file, such as the `--gcc-toolchain=...` flag.
3. Create a symlink to clang in the same directory that `clang++` and the config file exist in: `ln -s clang++ <name>-clang++`, again replacing `<name>` with the name of your config file.
4. When invoking clang with the symlink, clang will automatically apply the configuration file. Now use `<name>-clang++` in the build process of AdaptiveCpp instead of clang++, e.g. use the cmake flags `-DCMAKE_CXX_COMPILER=<name>-clang++ -DCLANG_EXECUTABLE_PATH=/path/to/<name>-clang++`.




