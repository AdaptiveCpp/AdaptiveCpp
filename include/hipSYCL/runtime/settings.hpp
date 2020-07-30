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

#include <optional>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
namespace hipsycl {
namespace rt {

enum class setting { debug_level };

template <setting S> struct setting_trait {};

#define HIPSYCL_RT_MAKE_SETTING_TRAIT(S, string_identifier, setting_type)      \
  template <> struct setting_trait<S> {                                        \
    using type = setting_type;                                                 \
    static constexpr const char *str = string_identifier;                      \
  };

HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::debug_level, "debug_level", int)

class settings
{
public:
  template <setting S>
  typename setting_trait<S>::type
  get_or_default(typename setting_trait<S>::type default_value) const {
    if (has_setting<S>()) {
      return get<S>();
    }
    return default_value;
  }

  template <setting S> typename setting_trait<S>::type get() const {
    if constexpr(S == setting::debug_level){
      return _debug_level.value();
    }
    return typename setting_trait<S>::type{};
  }

  template <setting S> bool has_setting() const {
    return get_optional<S>().has_value();
  }

  settings() {
    int default_debug_level = 1;
#ifdef HIPSYCL_DEBUG_LEVEL
    default_debug_level = HIPSYCL_DEBUG_LEVEL;
#endif
    _debug_level = get_environment_variable_or_default<setting::debug_level>(
        default_debug_level);
  }

private:
  template <setting S, class T>
  T get_environment_variable_or_default(const T &default_value) {
    const char *env = std::getenv(get_environment_variable_name<S>().c_str());
    if (env) {
      
      T val;
      std::stringstream sstr{std::string{env}};
      sstr >> val;

      if (sstr.fail() || sstr.bad())
        return default_value;
      return val;
    }
    return default_value;
  }

  template <setting S> std::string get_environment_variable_name() {
    std::string id = setting_trait<S>::str;
    std::transform(id.begin(), id.end(), id.begin(), ::toupper);
    return "HIPSYCL_"+id;
  }
  
  template <setting S>
  const std::optional<typename setting_trait<S>::type>& get_optional(setting p) const {
    if constexpr(S == setting::debug_level){
      return _debug_level;
    }
    return typename setting_trait<S>::type{};
  }

  std::optional<int> _debug_level;
};

}
}

#endif
