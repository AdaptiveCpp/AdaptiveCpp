/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include "../instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <CL/opencl.hpp>

namespace hipsycl {
namespace rt {

// map between the SYCL timestamp and the OpenCL symbol
template<class TimeStamp>
struct ocl_profiling_info_map;

// submission timestamp in hipSYCL matches what OpenCL calls the QUEUED time
template<>
struct ocl_profiling_info_map<instrumentations::submission_timestamp>
{
  static constexpr cl_profiling_info value = CL_PROFILING_COMMAND_QUEUED;
  static constexpr const char * name = "CL_PROFILING_COMMAND_QUEUED";
};

template<>
struct ocl_profiling_info_map<instrumentations::execution_start_timestamp>
{
  static constexpr cl_profiling_info value = CL_PROFILING_COMMAND_START;
  static constexpr const char *name = "CL_PROFILING_COMMAND_START";
};

template<>
struct ocl_profiling_info_map<instrumentations::execution_finish_timestamp>
{
  static constexpr cl_profiling_info value = CL_PROFILING_COMMAND_END;
  static constexpr const char *name = "CL_PROFILING_COMMAND_END";
};

// Asynchronous timestamp used for command queued/submitted/start/finish metric
template <class Base> class ocl_async_timestamp : public Base {
public:
  ocl_async_timestamp(cl::Event evt) : _evt(evt) {}

  static constexpr cl_profiling_info clInfoValue =
      ocl_profiling_info_map<Base>::value;

  profiler_clock::time_point get_time_point() const override
  {
    cl_ulong stamp = 0;
    cl_int err = _evt.getProfilingInfo(clInfoValue, &stamp);
    if (err != CL_SUCCESS) {
      print_error(__acpp_here(),
        error_info{"could not get profiling info " +
            std::string(ocl_profiling_info_map<Base>::name),
          error_code{"CL", static_cast<int>(err)}});
    }
    return profiler_clock::time_point{profiler_clock::duration{stamp}};
  }

  // wait for associated event to complete
  void wait() const override
  {
    _evt.wait();
  };

private:
  cl::Event _evt;
};

using ocl_submission_timestamp =
    ocl_async_timestamp<instrumentations::submission_timestamp>;
using ocl_execution_start_timestamp =
    ocl_async_timestamp<instrumentations::execution_start_timestamp>;
using ocl_execution_finish_timestamp =
    ocl_async_timestamp<instrumentations::execution_finish_timestamp>;

struct ocl_async_profiling {
  std::shared_ptr<ocl_submission_timestamp> submission_time;
  std::shared_ptr<ocl_execution_start_timestamp> start_time;
  std::shared_ptr<ocl_execution_finish_timestamp> finish_time;
};

} // namespace rt
} // namespace hipsycl

