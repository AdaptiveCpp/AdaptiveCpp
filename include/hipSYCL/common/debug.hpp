/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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

#ifndef HIPSYCL_DEBUG_HPP
#define HIPSYCL_DEBUG_HPP

#define HIPSYCL_DEBUG_LEVEL_NONE 0
#define HIPSYCL_DEBUG_LEVEL_ERROR 1
#define HIPSYCL_DEBUG_LEVEL_WARNING 2
#define HIPSYCL_DEBUG_LEVEL_INFO 3
#define HIPSYCL_DEBUG_LEVEL_VERBOSE 4

#ifndef HIPSYCL_DEBUG_LEVEL
#define HIPSYCL_DEBUG_LEVEL HIPSYCL_DEBUG_LEVEL_WARNING
#endif

#include <iostream>
#include <cstdlib>
#include <unistd.h>

#ifndef HIPSYCL_COMPILER_COMPONENT
#include "hipSYCL/runtime/application.hpp"
#endif

namespace hipsycl {
namespace common {

class output_stream {
public:
  static output_stream &get() {
    static output_stream ostr;
    return ostr;
  }

  std::ostream &get_stream() const { return _output_stream; }
  int get_debug_level() const { return _debug_level; }

private:

  output_stream()
  : _debug_level {HIPSYCL_DEBUG_LEVEL}, _output_stream{std::cout} {
#if !defined(HIPSYCL_COMPILER_COMPONENT) && !defined(HIPSYCL_TOOL_COMPONENT)
    _debug_level =
        rt::application::get_settings().get<rt::setting::debug_level>();
#else
    
    auto process_env = [this](const char* e) {
      if (std::string{e}.find_first_not_of("0123456789") ==
          std::string::npos) {
        _debug_level = std::stoi(std::string{e});
      }
    };
    
    if (const char *env = std::getenv("ACPP_DEBUG_LEVEL")) {
      process_env(env);
    } else if (const char *env = std::getenv("HIPSYCL_DEBUG_LEVEL")){
      process_env(env);
    }
#endif

    _isatty = isatty(fileno(stdout)) && isatty(fileno(stderr));
    if (!_isatty)
      // define HIPSYCL_DEBUG_NOCOLOR
      ;
  }

  int _debug_level;
  std::ostream& _output_stream;
  int _isatty;
};

}
}
#ifndef HIPSYCL_COMPILER_COMPONENT
#define HIPSYCL_DEBUG_STREAM(level, prefix)                                    \
  if (level > ::hipsycl::common::output_stream::get().get_debug_level())       \
    ;                                                                          \
  else ::hipsycl::common::output_stream::get().get_stream() << prefix
#else
#define HIPSYCL_DEBUG_STREAM(level, prefix)                                    \
  if (level > ::hipsycl::common::output_stream::get().get_debug_level())       \
    ;                                                                          \
  else llvm::outs() << prefix
#endif

#ifdef HIPSYCL_DEBUG_NOCOLOR
#define HIPSYCL_DEBUG_PREFIX_ERROR   "[AdaptiveCpp Error] "
#define HIPSYCL_DEBUG_PREFIX_WARNING "[AdaptiveCpp Warning] "
#define HIPSYCL_DEBUG_PREFIX_INFO    "[AdaptiveCpp Info] "
#else
#define HIPSYCL_DEBUG_PREFIX_ERROR   "\033[1;31m[AdaptiveCpp Error] \033[0m"
#define HIPSYCL_DEBUG_PREFIX_WARNING "\033[;35m[AdaptiveCpp Warning] \033[0m"
#define HIPSYCL_DEBUG_PREFIX_INFO    "\033[;32m[AdaptiveCpp Info] \033[0m"
#endif

#define HIPSYCL_DEBUG_ERROR \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_ERROR, \
                      HIPSYCL_DEBUG_PREFIX_ERROR)


#define HIPSYCL_DEBUG_WARNING \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_WARNING, \
                      HIPSYCL_DEBUG_PREFIX_WARNING)


#define HIPSYCL_DEBUG_INFO \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_INFO, \
                      HIPSYCL_DEBUG_PREFIX_INFO)

#define HIPSYCL_DEBUG_EXECUTE(level, content) \
  if(level <= ::hipsycl::common::output_stream::get().get_debug_level()) \
  {\
    content;\
  }

#define HIPSYCL_DEBUG_EXECUTE_VERBOSE(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_VERBOSE, content)
#define HIPSYCL_DEBUG_EXECUTE_INFO(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_INFO, content)
#define HIPSYCL_DEBUG_EXECUTE_WARNING(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_WARNING, content)
#define HIPSYCL_DEBUG_EXECUTE_ERROR(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_ERROR, content)
#endif
