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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include <fstream>
#include <type_traits>

namespace hipsycl::common::db {

namespace {

template <class T>
void print_key_value_pair(std::ostream &ostr, const std::string &key,
                          const T &val, int indentation_level) {
  for(int i = 0; i < indentation_level; ++i)
    ostr << "  ";
  ostr << key << ": " << val << std::endl;
}

template <class ArrayT>
void print_array(std::ostream &ostr, const std::string &name, const ArrayT &a,
                 const std::string &element_type_name, int indentation_level) {
  print_key_value_pair(ostr, name, "<array>", indentation_level);
  for(int i = 0; i < a.size(); ++i) {
    if constexpr(std::is_fundamental_v<typename ArrayT::value_type>)
      print_key_value_pair(ostr, std::to_string(i), a[i], indentation_level+1);
    else {
      print_key_value_pair(ostr, std::to_string(i), "<" + element_type_name + ">",
                         indentation_level + 1);
      a[i].dump(ostr, indentation_level + 2);
    }
  }
}
}

void kernel_arg_value_statistics::dump(std::ostream& ostr, int indentation_level) const {
  print_key_value_pair(ostr, "value", value, indentation_level);
  print_key_value_pair(ostr, "count", count, indentation_level);
  print_key_value_pair(ostr, "last_used", last_used, indentation_level);
}

void kernel_arg_entry::dump(std::ostream& ostr, int indentation_level) const {
  print_array(ostr, "common_values", common_values, "arg_statistics", indentation_level);
  print_array(ostr, "was_specialized", was_specialized, "bool", indentation_level);
}

void kernel_entry::dump(std::ostream& ostr, int indentation_level) const {
  print_key_value_pair(ostr, "num_registered_invocations",
                       num_registered_invocations, indentation_level);
  print_array(ostr, "kernel_args", kernel_args, "arg_entry", indentation_level);
}

void appdb_data::dump(std::ostream& ostr, int indentation_level) const {
  print_key_value_pair(ostr, "content_version", content_version, indentation_level);
  print_key_value_pair(ostr, "kernels", "<map>", indentation_level);

  for(const auto& entry : kernels) {
    std::string kernel_name = std::to_string(entry.first[0]);
    for(int i = 1; i < entry.first.size(); ++i)
      kernel_name += "-" + std::to_string(entry.first[i]);

    print_key_value_pair(ostr, kernel_name, "<kernel-entry>", indentation_level+1);
    
    entry.second.dump(ostr, indentation_level+2);
  }
}

appdb::appdb(const std::string& db_path) 
: _db_path{db_path}, _lock{0}, _was_modified{false} {

  if(filesystem::exists(db_path)) {
    std::ifstream file{_db_path, std::ios::in | std::ios::binary | std::ios::ate};
    if (!file.is_open())
      return;
    std::streamsize file_size = file.tellg();
    
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_content;
    file_content.resize(file_size);
    file.read(reinterpret_cast<char *>(file_content.data()), file_size);

    _data = msgpack::unpack<appdb_data>(file_content);
  }
}

appdb::~appdb() {
  if(_was_modified) {
    ++_data.content_version;

    auto data = msgpack::pack(_data);
    std::string data_string;
    data_string.resize(data.size());
    std::memcpy(data_string.data(), data.data(), data.size());

    common::filesystem::atomic_write(_db_path, data_string);
  }
}

}