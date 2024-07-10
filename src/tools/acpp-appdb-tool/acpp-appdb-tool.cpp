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
#include <iostream>
#include <string>

#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/appdb.hpp"


void usage() {
  std::cout << "Usage: acpp-appdb-tool </path/to/app.db or /full/path/to/executable> <-p|-c>\n"
            << "  -p: Print content of app db\n"
            << "  -c: Clear this app db" << std::endl;
}

bool is_appdb(const std::string& path) {
  std::string ending =
      "/" + hipsycl::common::filesystem::persistent_storage::get()
                .generate_app_db_filename();
  return path.find(ending) == path.size() - ending.size();
}

void print_content(const std::string& path) {
  hipsycl::common::db::appdb db{path};
  db.read_access([](const hipsycl::common::db::appdb_data& data){
    data.dump(std::cout);
  });
}

int main(int argc, char** argv) {
  if(argc != 3) {
    usage();
    return -1;
  }

  std::string path = argv[1];
  std::string appdb_path;
  
  if(is_appdb(path)) {
    appdb_path = path;
  } else {
#ifndef _WIN32
    appdb_path =
        hipsycl::common::filesystem::persistent_storage::get()
            .generate_appdb_path(hipsycl::common::filesystem::absolute(path));
#else
    appdb_path =
        hipsycl::common::filesystem::persistent_storage::get()
            .generate_appdb_path("");
#endif
  }

  std::string command = argv[2];

  if(command == "-p")
    print_content(appdb_path);
  else if(command == "-c")
    hipsycl::common::filesystem::remove(appdb_path);
  else {
    usage();
    return -1;
  }


  return 0;
}
