#pragma once
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"

namespace hipsycl::fp16 {


ACPP_UNIVERSAL_TARGET // So that CUDA calls are possible
static inline float fp32_from_bits(__acpp_uint32 w) {
#if defined(__OPENCL_VERSION__)
	return as_float(w);
#elif ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA == 1 && !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
	return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
	return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return _CopyFloatFromInt32((__int32) w);
#else
	union {
		__acpp_uint32 as_bits;
		float as_value;
	} fp32 = { w };
	return fp32.as_value;
#endif
}

ACPP_UNIVERSAL_TARGET // So that CUDA calls are possible
static inline __acpp_uint32 fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
	return as_uint(f);
#elif ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA == 1 && !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
	return (__acpp_uint32) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
	return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return (__acpp_uint32) _CopyInt32FromFloat(f);
#else
	union {
		float as_value;
		__acpp_uint32 as_bits;
	} fp32 = { f };
	return fp32.as_bits;
#endif
}

ACPP_UNIVERSAL_TARGET // So that CUDA calls are possible
static inline double fp64_from_bits(__acpp_uint64 w) {
#if defined(__OPENCL_VERSION__)
	return as_double(w);
#elif ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA == 1 && !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
	return __longlong_as_double((long long) w);
#elif defined(__INTEL_COMPILER)
	return _castu64_f64(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return _CopyDoubleFromInt64((__int64) w);
#else
	union {
		__acpp_uint64 as_bits;
		double as_value;
	} fp64 = { w };
	return fp64.as_value;
#endif
}

ACPP_UNIVERSAL_TARGET // So that CUDA calls are possible
static inline __acpp_uint64 fp64_to_bits(double f) {
#if defined(__OPENCL_VERSION__)
	return as_ulong(f);
#elif ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA == 1 && !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
	return (__acpp_uint64) __double_as_longlong(f);
#elif defined(__INTEL_COMPILER)
	return _castf64_u64(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return (__acpp_uint64) _CopyInt64FromDouble(f);
#else
	union {
		double as_value;
		__acpp_uint64 as_bits;
	} fp64 = { f };
	return fp64.as_bits;
#endif
}

}
