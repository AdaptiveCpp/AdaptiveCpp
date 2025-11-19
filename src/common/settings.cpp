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

#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/settings.hpp"

#include <string>
#include <fstream>

namespace hipsycl {
namespace common::settings {

namespace {

std::unordered_map<std::string, std::string> parse_config_file(const std::string& filename){
  std::ifstream file{filename};
  std::unordered_map<std::string, std::string> result;
  if(file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      auto equal_pos = line.find("=");
      if(equal_pos != std::string::npos) {
        std::string key = line.substr(0, equal_pos);
        if(equal_pos+1 < line.length()) {
          result[key] = line.substr(equal_pos+1);
        }
      }
    }
  }

  return result;
}

void trim(std::string& str) {
  str.erase(0, str.find_first_not_of("\t\n\v\f\r "));
  str.erase(str.find_last_not_of("\t\n\v\f\r ") + 1);
}

bool default_allocation_tracking = true;

}

bool get_default_enable_allocation_tracking() {
  return __atomic_load_n(&default_allocation_tracking, __ATOMIC_RELAXED);
}

void force_default_enable_allocation_tracking(bool v) {
  __atomic_store_n(&default_allocation_tracking, v, __ATOMIC_RELAXED);
}

void settings_config_file::load_file(const std::string& filename) {
  auto kv_map = parse_config_file(filename);
  
  for(const auto& entry : kv_map) {
    std::string key = entry.first;
    std::string value = entry.second;
    trim(key);
    trim(value);
    _values[key] = value;
  }
}

settings_config_file::settings_config_file() {
  std::string app_filename, app_directory;
  common::filesystem::get_this_executable_path(&app_filename, &app_directory);
  if(!app_directory.empty()){
    std::string acpp_config_file =
        common::filesystem::join_path(app_directory, "acpp-config.cfg");
    if(common::filesystem::exists(acpp_config_file)) {
      load_file(acpp_config_file);
    }
    if(!app_filename.empty()) {
      std::string app_config_file = common::filesystem::join_path(
          app_directory, "acpp-config-" + app_filename + ".cfg");
      if(common::filesystem::exists(app_config_file)) {
        load_file(app_config_file);
      }
    }
  }
}

bool settings_config_file::retrieve_setting(const std::string& key, std::string& value) const {
  auto it = _values.find(key);
  if(it != _values.end()) {
    value = it->second;
    return true;
  }
  return false;
}

settings_config_file& settings_config_file::get() {
  static settings_config_file cfg;
  return cfg;
}


}
}
