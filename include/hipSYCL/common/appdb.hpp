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
#include <atomic>
#include <vector>
#include <string>
#include <ostream>

#include "msgpack/msgpack.hpp"

#include "hipSYCL/runtime/kernel_configuration.hpp"

namespace hipsycl::common::db {

struct kernel_arg_value_statistics {
  uint64_t value; // The kernel argument value
  uint64_t count; // How many times we have seen this value
  uint64_t last_used; // The number of the kernel invocation where this value
                      // was last used

  void dump(std::ostream& ostr, int indentation_level=0) const;

  template<class T>
  void pack(T &pack) {
    pack(value);
    pack(count);
    pack(last_used);
  }
};

struct kernel_arg_entry {
  static constexpr int max_tracked_values = 8;

  std::array<kernel_arg_value_statistics, max_tracked_values> common_values = {};
  std::array<bool, max_tracked_values> was_specialized = {};

  template<class T>
  void pack(T &pack) {
    pack(common_values);
    pack(was_specialized);
  }

  void dump(std::ostream& ostr, int indentation_level=0) const;
};

struct kernel_entry {

  template<class T>
  void pack(T &pack) {
    pack(kernel_args);
    pack(num_registered_invocations);
  }

  void dump(std::ostream& ostr, int indentation_level=0) const;

  std::vector<kernel_arg_entry> kernel_args;
  std::size_t num_registered_invocations;
};

struct appdb_data {
  std::size_t content_version = 0;

  std::unordered_map<rt::kernel_configuration::id_type, kernel_entry,
                     rt::kernel_id_hash>
      kernels;

  template<class T>
  void pack(T &pack) {
    pack(kernels);
    pack(content_version);
  }

  void dump(std::ostream& ostr, int indentation_level=0) const;
};


class appdb  {
public:
  // DO NOT FORGET TO INCREMENT THIS WHEN ADDING/REMOVING
  // FIELDS OR OTHERWISE CHANGING THE DATA LAYOUT!
  static const uint64_t format_version = 1;

  appdb(const std::string& db_path);
  ~appdb();

  template<class F>
  void read_access(F&& handler) const{
    read_lock lock {_lock};
    handler(_data);
  }

  template<class F>
  void read_write_access(F&& handler) {
    write_lock lock {_lock};
    handler(_data);
    _was_modified = true;
  }

private:
  
  struct write_lock {
  public:
    write_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = 0;
      while (!_op_counter.compare_exchange_strong(
          expected, -1, std::memory_order_release, std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    ~write_lock() {
      _op_counter.store(0, std::memory_order_release);
    }
  private:
    std::atomic<int>& _op_counter;
  };

  struct read_lock {
  public:
    read_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = std::max(0, _op_counter.load(std::memory_order_acquire));
      while (!_op_counter.compare_exchange_strong(
          expected, expected+1, std::memory_order_release,
          std::memory_order_relaxed)) {
        if(expected < 0)
          expected = 0;
      }
    }

    ~read_lock() {
      _op_counter.fetch_sub(1, std::memory_order_acq_rel);
    }
  private:
   std::atomic<int>& _op_counter;
  };

  mutable std::atomic<int> _lock;
  bool _was_modified;

  std::string _db_path;

  appdb_data _data;
};


}

#endif
