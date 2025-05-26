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

namespace hipsycl {
namespace compiler {

namespace {

std::string getRedistributablePackagePath() {
  const auto install_dir = common::filesystem::get_install_directory();
  return common::filesystem::join_path(install_dir,
                                       std::vector<std::string>{"lib", "hipSYCL", "ext"});
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
      llvm_redistributable_path, std::vector<std::string>{"bin", "llc"});

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
      llvm_redistributable_path, std::vector<std::string>{"bin", "lld"});

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
      llvm_redistributable_path, std::vector<std::string>{"bin", "opt"});

  if(common::filesystem::exists(opt_redistributable_path)) {
    path = opt_redistributable_path;
  } else {
    path = replacePathPlaceholders(ACPP_OPT_PATH);
  }

  return path;
}

std::string getRedistPackageBitcodePath(const std::string& backend) {
  return common::filesystem::join_path(getRedistributablePackagePath(),
                                       std::vector<std::string>{"bitcode", backend});
}

} // namespace compiler
} // namespace hipsycl
