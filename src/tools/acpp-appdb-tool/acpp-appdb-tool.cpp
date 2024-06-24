/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay and contributors
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
