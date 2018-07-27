# SYCU - an implementation of SYCL over NVIDIA CUDA/AMD HIP
The goal of the SYCU project is to develop a SYCL 1.2.1 implementation that is built upon NVIDIA CUDA/AMD HIP.
Essentially, SYCU is a SYCL wrapper around CUDA/HIP, i.e. your applications are compiled with nvcc (NVIDIA) or hcc (AMD) like regular CUDA/HIP applications.

## Rationale
Present solutions (OpenCL, SYCL, CUDA, HIP) require choosing either portability or access to latest hardware features.
For example, when writing an application in CUDA, because of vendor support by NVIDIA, latest hardware features of NVIDIA GPUs can be exploited and there are sophisticated debugging/profiling tools available.
However, the application is then inherently unportable and will only run on NVIDIA GPUs.
If portability is more important, OpenCL can be used instead, which is however not well supported by some vendors, and some newer features are not implemented by these vendors. Here, especially NVIDIA stands out as a trouble spot.

SYCU attempts to solve these problems by providing a SYCL interface to CUDA/HIP. This means that
* You program your applications against the SYCL interface, which is an open standard. There are several other SYCL implementations available that allow for an execution of SYCL-compliant code on any OpenCL device, from CPUs to GPUs and FPGAs. In other words, if you don't want to run your code on GPUs from a certain vendor anymore, if it's written in SYCL, you can easily run it on any other device.
* Since SYCU is effectively a CUDA/HIP wrapper, you have full vendor support from NVIDIA (and AMD). Consequently, all CUDA/HIP profiling debugging tools are expected to work with SYCU.
* Since SYCU code relies on compilation with nvcc/hcc, you can easily create optimized code paths for the latest GPUs, and all the latest features that are available in CUDA will also be available to you.

## Current state
SYCU is still in an early stage of development. At the moment, you should not expect to be able to use it for any meaningful applications.

Detailed state:
* Runtime API: Mostly done
* Memory management: WIP
* Device library: TBD


