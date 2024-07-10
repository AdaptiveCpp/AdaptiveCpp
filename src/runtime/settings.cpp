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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/string_utils.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/settings.hpp"

#include <string>

namespace hipsycl {
namespace rt {

std::istream &operator>>(std::istream &istr, scheduler_type &out) {
  std::string str;
  istr >> str;
  if (str == "direct")
    out = scheduler_type::direct;
  else if (str == "unbound")
    out = scheduler_type::unbound;
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

namespace {

void trim(std::string& str) {
  str.erase(0, str.find_first_not_of("\t\n\v\f\r "));
  str.erase(str.find_last_not_of("\t\n\v\f\r ") + 1);
}

bool is_number(const std::string& str){
  return str.find_first_not_of("0123456789") == std::string::npos;
}
}

visibility_mask_t::mapped_type parse_device_visibility_mask(const std::string& str) {
  visibility_mask_t::mapped_type device_visibility_conditions;

  std::vector<std::string> condition_substrings = common::split_by_delimiter(str, ',', false);
  for(auto& s : condition_substrings) {
    trim(s);
    device_visibility_condition current;
    auto components = common::split_by_delimiter(s, '.', true);
    for(auto& c : components)
      trim(c);

    if (components.size() == 1) {
      if(is_number(components[0])) {
        current.device_index_equality = std::stoi(components[0]);
      } else if(components[0] != "*") {
        current.device_name_match = components[0]; 
      }
    } else if (components.size() > 1) {
      if(is_number(components[0])) {
        current.platform_index_equality = std::stoi(components[0]);
      } else if(components[0] != "*") {
        current.platform_name_match = components[0]; 
      }
      
      if(is_number(components[1])) {
        current.device_index_equality = std::stoi(components[1]);
      } else if(components[1] != "*") {
        current.device_name_match = components[1];
      }
    }
    device_visibility_conditions.push_back(current);
  }
  
  return device_visibility_conditions;
}

bool device_matches(const visibility_mask_t::mapped_type &conditions,
                    int global_device_index, int platform_device_index,
                    int platform_index, const std::string &dev_name,
                    const std::string &platform_name) {
  if(conditions.empty())
    return true;
  
  // The logic is: All individual device visibility conditions are connected by or,
  // but the conditions within each condition are connected by and.
  for(const auto& c : conditions) {

    // If we are given conditions about the platform, use a device index relative to the
    // platform.
    int device_index =
        (c.platform_index_equality >= 0 || !c.platform_name_match.empty())
            ? platform_device_index
            : global_device_index;

    bool all_true = true;
    if(c.platform_index_equality >= 0 && (platform_index != c.platform_index_equality))
      all_true = false;
    if(c.device_index_equality >= 0 && (device_index != c.device_index_equality))
      all_true = false;
    if(!c.device_name_match.empty() && (dev_name.find(c.device_name_match) == std::string::npos))
      all_true = false;
    if(!c.platform_name_match.empty() && (platform_name.find(c.platform_name_match) == std::string::npos))
      all_true = false;
    if(all_true)
      return true;

  }
  return false;
}

bool device_matches(const visibility_mask_t &mask, backend_id backend,
                    int global_device_index, int platform_device_index,
                    int platform_index, const std::string &dev_name,
                    const std::string &platform_name) {
  auto it = mask.find(backend);
  if(it == mask.end())
    return true;

  return device_matches(it->second, global_device_index, platform_device_index,
                        platform_index, dev_name, platform_name);
}

bool has_device_visibility_mask(const visibility_mask_t& mask, backend_id backend) {
  auto it = mask.find(backend);
  if(it != mask.end()) {
    return it->second.size() > 0;
  }
  return false;
}

std::istream &operator>>(std::istream &istr, visibility_mask_t &out) {
  std::string str;
  istr >> str;
  // have to copy, as otherweise might be interpreted as failing, although everything is fine.
  std::istringstream istream{str};

  std::string backend_specific_substring;
  while(std::getline(istream, backend_specific_substring, ';')) {
    if(backend_specific_substring.empty())
      continue;
    
    std::size_t delimiter = backend_specific_substring.find(':');
    std::string name;
    if(delimiter != std::string::npos) {
      name = backend_specific_substring.substr(0, delimiter);
    } else {
      name = backend_specific_substring;
    }

    std::transform(name.cbegin(), name.cend(), name.begin(), ::tolower);

    rt::backend_id backend;
    if (name == "cuda") {
      backend = rt::backend_id::cuda;
    } else if (name == "hip") {
      backend = rt::backend_id::hip;
    } else if (name == "ze") {
      backend = rt::backend_id::level_zero;
    } else if (name == "omp") {
      // looking for this, even though we have to allow it always.
      backend = rt::backend_id::omp;
    } else if (name == "ocl" || name == "opencl") {
      backend = rt::backend_id::ocl;
    } else {
      istr.setstate(std::ios_base::failbit);
      // Don't use HIPSYCL_DEBUG_WARNING, it will cause recursive init error.
      std::cout << "'" << name << "' is not a known backend name." << std::endl;
      break;
    }

    if(delimiter != std::string::npos) {
      auto device_visibility_mask = parse_device_visibility_mask(
          backend_specific_substring.substr(delimiter + 1));
      out[backend] = device_visibility_mask;
    } else
      out[backend] = {};
  }
  return istr;
}

std::istream &operator>>(std::istream &istr, default_selector_behavior& out) {
  std::string str;
  istr >> str;
  if (str == "strict")
    out = default_selector_behavior::strict;
  else if (str == "multigpu")
    out = default_selector_behavior::multigpu;
  else if (str == "system")
    out = default_selector_behavior::system;
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

}
}