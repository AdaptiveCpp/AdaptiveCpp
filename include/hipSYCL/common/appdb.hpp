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

#ifndef HIPSYCL_COMMON_APP_DB_HPP
#define HIPSYCL_COMMON_APP_DB_HPP

#include <unordered_map>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>

#include "msgpack/msgpack.hpp"

#include "hipSYCL/runtime/kernel_configuration.hpp"

namespace hipsycl::common::db {

struct kernel_arg_entry {
  static constexpr int max_tracked_values = 8;

  std::array<uint64_t, max_tracked_values> common_values;
  std::array<uint64_t, max_tracked_values> common_values_count;
};

struct kernel_entry {
  template<class T>
  void msgpack(T &pack) {
    pack(kernel_args);
  }

  std::vector<kernel_arg_entry> kernel_args;
};

struct appdb_data {
  std::unordered_map<rt::kernel_configuration::id_type, kernel_entry,
                     rt::kernel_id_hash>
      kernels;

  template<class T>
  void msgpack(T &pack) {
    pack(kernels);
  }
};


class appdb  {
public:
  appdb(const std::string& db_path, bool read_only = false);
  ~appdb();

  template<class F>
  void access(F&& handler) {
    if (_read_only) {
      std::lock_guard<std::mutex> lock{_mutex};
      handler(_data);
    } else {
      handler(_data);
    }
  }
private:
  std::mutex _mutex;
  std::string _db_path;

  appdb_data _data;

  bool _read_only;
};


}

#endif
