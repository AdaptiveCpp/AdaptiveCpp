#ifndef SYCL_HPP
#define SYCL_HPP

#define __SYCU__

#if defined(__CUDA__)

#define SYCU_PLATFORM_CUDA
#define sycu_kernel __host__ __device__
#define sycu_device_only __device__
#define sycu_host __host__

#elif defined(__HCC__)
#define SYCU_PLATFORM_HIP
#define sycu_kernel __host__ __device__
#define sycu_device_only __device__
#define sycu_host __host__

#else

#define SYCU_PLATFORM_HOST

#define sycu_device
#define sycu_host
#endif

#include "sycl/version.hpp"
#include "sycl/types.hpp"

#endif

