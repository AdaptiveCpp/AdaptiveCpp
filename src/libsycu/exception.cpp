#include "CL/sycl/exception.hpp"
#include "CL/sycl/context.hpp"

namespace cl {
namespace sycl {

context exception::get_context() const
{
  return context{};
}

namespace detail {

void check_error(hipError_t e) {

  switch(e) {

  case hipSuccess:
    return;
  case hipErrorNotReady:
    // Not really an error, since it indicates that an operation is enqueued
    return;
  case hipErrorNoDevice:
    // This occurs when no devices are found. Do not throw in that case,
    // we want the user to treat the case when no devices are
    // available without try/catch
    return;
  case hipErrorInvalidContext:
    throw platform_error{"Input context is invalid", e};
  case hipErrorInvalidKernelFile:
    throw platform_error{"Invalid PTX",e};
  case hipErrorMemoryAllocation:
    throw memory_allocation_error{"Bad memory allocation",e};
  case hipErrorInitializationError:
    throw exception{"Initialization error", e};
  case hipErrorLaunchFailure:
    throw kernel_error{"An error occurred on the device while executing a kernel.", e};
  case hipErrorInvalidDevice:
    throw device_error{"Invalid device id", e};
  case hipErrorInvalidValue:
    throw runtime_error{"One or more of the parameters passed to the API "
                        "call is NULL or not in an acceptable range.",e};
  case hipErrorInvalidDevicePointer:
    throw invalid_parameter_error{"Invalid device pointer",e};
  case hipErrorInvalidMemcpyDirection:
    throw invalid_parameter_error{"Invalid memcpy direction",e};
  case hipErrorUnknown:
    throw exception{"Unknown HIP error",e};
  case hipErrorInvalidResourceHandle:
    throw invalid_object_error{"Invalid event or queue",e};
  case hipErrorRuntimeMemory:
    throw device_error{"HSA memory error",e};
  case hipErrorRuntimeOther:
    throw device_error{"HSA error",e};
  case hipErrorHostMemoryAlreadyRegistered:
    throw runtime_error{"Could not lock page-locked memory",e};
  case hipErrorHostMemoryNotRegistered:
    throw runtime_error{"Could not unlock non-page-locked memory",e};
  case hipErrorMapBufferObjectFailed:
    throw runtime_error{"IPC memory attach failed from ROCr",e};
  default:
    throw exception{"Unknown error occured", e};
  }

}

}

}
}
