# AdaptiveCpp installation instructions for SPIR-V/OpenCL

You will need an OpenCL implementation, and the OpenCL icd loader. The OpenCL library can be specified using `cmake -DOpenCL_LIBRARY=/path/to/libOpenCL.so`.

The OpenCL backend can be enabled using `cmake -DWITH_OPENCL_BACKEND=ON` when building AdaptiveCpp.
In order to run code successfully on an OpenCL device, it must support SPIR-V ingestion and the Intel USM (unified shared memory) extension. In a degraded mode, devices supporting OpenCL fine-grained system SVM (shared virtual memory) may work as well.

