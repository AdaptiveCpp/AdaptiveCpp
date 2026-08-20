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
#include "hipSYCL/runtime/vk/vk_profiling.hpp"

namespace hipsycl {
namespace rt {

calibrated_timestamps vk_get_calibrated_timestamps(const vk::raii::Device &dev,
                                                   bool use_khr) {
  std::array<vk::CalibratedTimestampInfoEXT, 2> timestamp_infos = {
      vk::CalibratedTimestampInfoEXT(vk::TimeDomainEXT::eClockMonotonic),
      vk::CalibratedTimestampInfoEXT(vk::TimeDomainEXT::eDevice)};

  auto time_domains = use_khr ? dev.getCalibratedTimestampsKHR(timestamp_infos)
                              : dev.getCalibratedTimestampsEXT(timestamp_infos);
  auto &timestamps = time_domains.first; // Drop max deviation
  return calibrated_timestamps{timestamps[0], timestamps[1]};
}

uint64_t vk_get_query_pool_result(const vk::raii::QueryPool &query_pool) {
  // Get timestamp data as 64-bit int rather than 32-bit default.
  auto [result, data] = query_pool.getResults<uint64_t>(
      0, 1, sizeof(uint64_t), 0,
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

  switch (result) {
  case vk::Result::eSuccess:
    break;
  case vk::Result::eNotReady:
    print_error(__acpp_here(),
                error_info{"vkGetQueryPoolResults returned VK_NOT_READY"});
    break;
  default:
    print_error(
        __acpp_here(),
        error_info{"vkGetQueryPoolResults returned an unexpected error"});
    break;
  }

  return data[0];
}

vk::raii::QueryPool vk_create_query_pool(const vk::raii::Device &dev) {
  vk::QueryPoolCreateInfo query_pool_info{};
  query_pool_info.queryType = vk::QueryType::eTimestamp;
  query_pool_info.queryCount = 1;
  return dev.createQueryPool(query_pool_info);
}

} // namespace rt
} // namespace hipsycl
