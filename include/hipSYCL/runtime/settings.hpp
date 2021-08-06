/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_RT_SETTINGS_HPP
#define HIPSYCL_RT_SETTINGS_HPP

#include "hipSYCL/runtime/device_id.hpp"
#include <ios>
#include <optional>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
namespace hipsycl {
namespace rt {

enum class scheduler_type { direct, unbound };
enum class default_selector_behavior { strict, multigpu, system };

std::istream &operator>>(std::istream &istr, scheduler_type &out);
std::istream &operator>>(std::istream &istr, std::vector<rt::backend_id> &out);
std::istream &operator>>(std::istream &istr, default_selector_behavior& out);

enum class setting {
  debug_level,
  scheduler_type,
  visibility_mask,
  dag_req_optimization_depth,
  mqe_lane_statistics_max_size,
  mqe_lane_statistics_decay_time_sec,
  default_selector_behavior
};

template <setting S> struct setting_trait {};

#define HIPSYCL_RT_MAKE_SETTING_TRAIT(S, string_identifier, setting_type)      \
  template <> struct setting_trait<S> {                                        \
    using type = setting_type;                                                 \
    static constexpr const char *str = string_identifier;                      \
  };

HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::debug_level, "debug_level", int)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::scheduler_type, "rt_scheduler", scheduler_type)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::visibility_mask, "visibility_mask", std::vector<rt::backend_id>)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::dag_req_optimization_depth,
                              "rt_dag_req_optimization_depth", std::size_t);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::mqe_lane_statistics_max_size,
                              "rt_mqe_lane_statistics_max_size", std::size_t);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::mqe_lane_statistics_decay_time_sec,
                              "rt_mqe_lane_statistics_decay_time_sec", double);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::default_selector_behavior,
                              "default_selector_behavior", default_selector_behavior);

class settings
{
public:

  template <setting S> typename setting_trait<S>::type get() const {
    if constexpr(S == setting::debug_level){
      return _debug_level;
    } else if constexpr (S == setting::scheduler_type) {
      return _scheduler_type;
    } else if constexpr (S == setting::visibility_mask) {
      return _visibility_mask;
    } else if constexpr (S == setting::dag_req_optimization_depth) {
      return _dag_requirement_optimization_depth;
    } else if constexpr (S == setting::mqe_lane_statistics_max_size) {
      return _mqe_lane_statistics_max_size;
    } else if constexpr (S == setting::mqe_lane_statistics_decay_time_sec) {
      return _mqe_lane_statistics_decay_time_sec;
    } else if constexpr (S == setting::default_selector_behavior) {
      return _default_selector_behavior;
    }
    return typename setting_trait<S>::type{};
  }

  settings() {
    int default_debug_level = 2;
#ifdef HIPSYCL_DEBUG_LEVEL
    default_debug_level = HIPSYCL_DEBUG_LEVEL;
#endif
    _debug_level = get_environment_variable_or_default<setting::debug_level>(
        default_debug_level);
    _scheduler_type =
        get_environment_variable_or_default<setting::scheduler_type>(
            scheduler_type::unbound);
    _visibility_mask =
        get_environment_variable_or_default<setting::visibility_mask>(
            std::vector<rt::backend_id>{});
    _dag_requirement_optimization_depth = get_environment_variable_or_default<
        setting::dag_req_optimization_depth>(10);
    _mqe_lane_statistics_max_size = get_environment_variable_or_default<
        setting::mqe_lane_statistics_max_size>(100);
    _mqe_lane_statistics_decay_time_sec = get_environment_variable_or_default<
        setting::mqe_lane_statistics_decay_time_sec>(10.0);
    _default_selector_behavior =
        get_environment_variable_or_default<setting::default_selector_behavior>(
            default_selector_behavior::strict);
  }

private:
  template <setting S, class T>
  T get_environment_variable_or_default(const T &default_value) {
    const char *env = std::getenv(get_environment_variable_name<S>().c_str());
    if (env) {
      
      T val;
      std::stringstream sstr{std::string{env}};
      sstr >> val;

      if (sstr.fail() || sstr.bad()) {
        std::cerr << "hipSYCL prelaunch: Could not parse value of environment "
                     "variable: "
                  << get_environment_variable_name<S>() << std::endl;
        return default_value;
      }
      return val;
    }
    return default_value;
  }

  template <setting S> std::string get_environment_variable_name() {
    std::string id = setting_trait<S>::str;
    std::transform(id.begin(), id.end(), id.begin(), ::toupper);
    return "HIPSYCL_"+id;
  }

  int _debug_level;
  scheduler_type _scheduler_type;
  std::vector<rt::backend_id> _visibility_mask;
  std::size_t _dag_requirement_optimization_depth;
  std::size_t _mqe_lane_statistics_max_size;
  double _mqe_lane_statistics_decay_time_sec;
  default_selector_behavior _default_selector_behavior;
};

}
}

#endif
