/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_COMMON_FILESYSTEM_HPP
#define HIPSYCL_COMMON_FILESYSTEM_HPP

#include <string>
#include <vector>

namespace hipsycl {
namespace common {

namespace filesystem {

std::string get_install_directory();

std::string join_path(const std::string& base, const std::string& extra);

std::string
join_path(const std::string &base,
          const std::vector<std::string> &additional_components);

std::vector<std::string> list_regular_files(const std::string &directory);
std::vector<std::string> list_regular_files(const std::string &directory,
                                            const std::string &extension);

/// Writes data atomically to filename
bool atomic_write(const std::string& filename, const std::string& data);

/// Removes a file, returns true if successful.
bool remove(const std::string &filename);

class tuningdb {
public:
  static tuningdb& get() {
    static tuningdb t;
    return t;
  }

  const std::string& get_base_dir() const {
    return _base_dir;
  }

  const std::string& get_this_app_dir() const {
    return _this_app_dir;
  }

  const std::string& get_jit_cache_dir() const {
    return _jit_cache_dir;
  }
private:
  tuningdb();

  std::string _base_dir;
  std::string _this_app_dir;
  std::string _jit_cache_dir;
};


}

}
}

#endif
