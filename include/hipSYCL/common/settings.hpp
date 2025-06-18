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
#ifndef ACPP_COMMON_SETTINGS_HPP
#define ACPP_COMMON_SETTINGS_HPP


#include <string>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "export.hpp"


namespace hipsycl {
namespace common::settings {


namespace detail {

inline std::string
generate_configuration_identifier(const std::string &name,
                                  bool legacy_prefix = false) {
  std::string capitalized_name = name;

  std::transform(capitalized_name.begin(), capitalized_name.end(),
                 capitalized_name.begin(), ::toupper);
  if(legacy_prefix)
    return "HIPSYCL_"+capitalized_name;
  else
    return "ACPP_"+capitalized_name;
}

}

class ACPP_COMMON_EXPORT settings_config_file {
public:
  static settings_config_file& get();
  bool retrieve_setting(const std::string& key, std::string& value) const;
private:
  settings_config_file();
  void load_file(const std::string& filename);
  std::unordered_map<std::string, std::string> _values;
};

template<class T>
bool try_retrieve_environment_variable(const std::string& name, T& out) {
  std::string env_name =
      detail::generate_configuration_identifier(name, false);
  std::string legacy_env_name =
      detail::generate_configuration_identifier(name, true);

  std::string env;
  if (const char *env_value =
          std::getenv(env_name.c_str())) {
    env = std::string{env_value};
  } else if (const char *env_value =
          std::getenv(legacy_env_name.c_str())) {
    env = std::string{env_value};
  }
  
  if (!env.empty()) {
    
    T val;
    std::stringstream sstr{std::string{env}};
    sstr >> val;

    if (sstr.fail() || sstr.bad()) {
      std::cerr << "AdaptiveCpp settings parsing: Could not parse value of environment "
                    "variable: "
                << env_name << std::endl;
      return false;
    }
    out = val;
    return true;
  }
  return false;
}

template <class T>
bool try_retrieve_settings_variable(const std::string& name, T& out) {

  if(try_retrieve_environment_variable(name, out))
    return true;
  
  // try cfg file
  std::string var_name =
    detail::generate_configuration_identifier(name);
  std::string value_string;

  if(settings_config_file::get().retrieve_setting(var_name, value_string)) {

    T val;
    std::istringstream sstr{std::string{value_string}};
    sstr >> val;

    if (sstr.fail() || sstr.bad()) {
      std::cerr << "AdaptiveCpp settings parsing: Could not parse value of config file "
                    "entry: "
                << var_name << std::endl;
      return false;
    }
    out = val;
    return true;
  }
  
  return false;
}


}
}

#endif
