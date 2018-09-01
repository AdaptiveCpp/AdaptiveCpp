/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CL/sycl/exception.hpp"
#include "CL/sycl/context.hpp"


namespace cl {
namespace sycl {

context exception::get_context() const
{
  // ToDo That's not entirely correct, as different contexts
  // can have different devices associated with them
  return context{};
}

namespace detail {

void check_error(hipError_t e) {

  switch(e)
  {
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
