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

#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/common/filesystem.hpp"

namespace hipsycl {
namespace compiler {

namespace {

std::string getRedistributablePackagePath() {
  const auto install_dir = common::filesystem::get_lib_directory();
  return common::filesystem::join_path(install_dir,
                                       std::vector<std::string>{"hipSYCL", "ext"});
}

std::string getLLVMRedistributablePackagePath() {
  std::string RedistPkg = getRedistributablePackagePath();
  return common::filesystem::join_path(RedistPkg, "llvm");
}

std::string replacePathPlaceholders(std::string path) {
  auto pos = path.find("$ACPP_PATH");
  while (pos != std::string::npos) {
    const auto install_dir = common::filesystem::get_install_directory();
    path.replace(pos, std::string_view("$ACPP_PATH").size(), install_dir);
    pos = path.find("$ACPP_PATH");
  }
  return path;
}

}

std::string getClangPath() {
  static std::string path;
  if(!path.empty())
    return path;
  else
    path = replacePathPlaceholders(ACPP_CLANG_PATH);
  
  return path;
}

std::string getLLCPath() {
  static std::string path;
  if(!path.empty())
    return path;
  
  std::string llvm_redistributable_path = getLLVMRedistributablePackagePath();
  std::string llc_redistributable_path = common::filesystem::join_path(
      llvm_redistributable_path, std::vector<std::string>{"bin", ACPP_LLC_NAME});

  if(common::filesystem::exists(llc_redistributable_path)) {
    path = llc_redistributable_path;
  } else {
    path = replacePathPlaceholders(ACPP_LLC_PATH);
  }

  return path;
}

std::string getLLDPath() {
  static std::string path;
  if(!path.empty())
    return path;
  
  std::string llvm_redistributable_path = getLLVMRedistributablePackagePath();
  std::string lld_redistributable_path = common::filesystem::join_path(
      llvm_redistributable_path, std::vector<std::string>{"bin", ACPP_LLD_NAME});

  if(common::filesystem::exists(lld_redistributable_path)) {
    path = lld_redistributable_path;
  } else {
    path = replacePathPlaceholders(ACPP_LLD_PATH);
  }

  return path;
}

std::string getOptPath() {
  static std::string path;
  if(!path.empty())
    return path;
  
  std::string llvm_redistributable_path = getLLVMRedistributablePackagePath();
  std::string opt_redistributable_path = common::filesystem::join_path(
      llvm_redistributable_path, std::vector<std::string>{"bin", ACPP_OPT_NAME});

  if(common::filesystem::exists(opt_redistributable_path)) {
    path = opt_redistributable_path;
  } else {
    path = replacePathPlaceholders(ACPP_OPT_PATH);
  }

  return path;
}

std::string getBitcodePath() {
#ifndef _WIN32
  return common::filesystem::join_path(common::filesystem::get_lib_directory(),
                                    std::vector<std::string>{"hipSYCL", "bitcode"});
#else
  static std::string bitcode_dir;
  if(bitcode_dir.empty()) {
    std::vector<std::string> candidates;
    // On Windows, lib_dir might be either bin/ or lib/ since libraries there might
    // be put in bin/ directory.
    std::string lib_dir = common::filesystem::get_lib_directory();
    candidates.emplace_back(lib_dir);
    candidates.emplace_back(common::filesystem::join_path(lib_dir,
      std::vector<std::string>{"..", "bin"}));
    candidates.emplace_back(common::filesystem::join_path(lib_dir,
      std::vector<std::string>{"..", "lib"}));
    for(const auto& candidate_root : candidates) {
      std::string candidate_bitcode_dir = common::filesystem::join_path(
        candidate_root, std::vector<std::string>{"hipSYCL", "bitcode"});
      if(common::filesystem::exists(candidate_bitcode_dir)) {
        std::error_code error;
        auto file_list = common::filesystem::list_regular_files(candidate_bitcode_dir, error);

        auto includes_bitcode_files = [](const std::vector<std::string>& filenames){
          for(const auto& f : filenames) {
            if(f.find(".bc") != std::string::npos)
              return true;
          }
          return false;
        };

        if(includes_bitcode_files(file_list)) {
          bitcode_dir = candidate_bitcode_dir;
          return bitcode_dir;
        }
      }
    }
    
  }
  return bitcode_dir;
#endif  
}

std::string getRedistPackageBitcodePath(const std::string& backend) {
  return common::filesystem::join_path(getRedistributablePackagePath(),
                                       std::vector<std::string>{"bitcode", backend});
}

} // namespace compiler
} // namespace hipsycl
