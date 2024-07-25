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
#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
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
  print_array(ostr, "retained_argument_indices", retained_argument_indices,
              "int", indentation_level);
  print_array(ostr, "kernel_args", kernel_args, "arg_entry", indentation_level);
  print_key_value_pair(ostr, "first_invocation_run",
                       first_iads_invocation_run, indentation_level);
}

void binary_entry::dump(std::ostream& ostr, int indentation_level) const {
  print_key_value_pair(ostr, "jit_cache_filename", jit_cache_filename,
                       indentation_level);
}

void appdb_data::dump(std::ostream& ostr, int indentation_level) const {
  print_key_value_pair(ostr, "content_version", content_version, indentation_level);
  
  auto get_id_string = [](const rt::kernel_configuration::id_type& id) {
    std::string kernel_name = std::to_string(id[0]);
    for(int i = 1; i < id.size(); ++i)
      kernel_name += "." + std::to_string(id[i]); 
    return kernel_name;
  };
  
  print_key_value_pair(ostr, "kernels", "<map>", indentation_level);

  for(const auto& entry : kernels) {
    std::string kernel_name = get_id_string(entry.first);
    print_key_value_pair(ostr, kernel_name, "<kernel-entry>", indentation_level+1);
    entry.second.dump(ostr, indentation_level+2);
  }

  print_key_value_pair(ostr, "binaries", "<map>", indentation_level);

  for(const auto& entry : binaries) {
    std::string binary_name = get_id_string(entry.first);
    print_key_value_pair(ostr, binary_name, "<binary-entry>", indentation_level+1);
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