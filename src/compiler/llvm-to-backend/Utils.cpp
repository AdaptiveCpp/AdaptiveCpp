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

#include <llvm/Support/Program.h>

#ifdef _WIN32
#include <llvm/Support/FileSystem.h>
#endif

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

std::string getLibSleefDir() {
  static std::string path;
  if (!path.empty())
    return path;

  const auto lib_path = common::filesystem::get_lib_directory();

#ifdef SLEEF_AVAILABLE
  std::string lib_sleef_redistributable_path =
      common::filesystem::join_path(lib_path, LIB_SLEEF_NAME);
  std::string lib_sleef_path = common::filesystem::join_path(LIB_SLEEF_DIR, LIB_SLEEF_NAME);

  if (common::filesystem::exists(lib_sleef_redistributable_path)) {
    path = lib_path;
  } else if (common::filesystem::exists(lib_sleef_path)) {
    path = replacePathPlaceholders(LIB_SLEEF_DIR);
  }
#endif

  return path;
}

std::string getLibAmathDir() {
  static std::string path;
  if (!path.empty())
    return path;

  const auto lib_path = common::filesystem::get_lib_directory();

#ifdef AMATH_AVAILABLE
  std::string lib_amath_redistributable_path =
      common::filesystem::join_path(lib_path, LIB_AMATH_NAME);
  std::string lib_amath_path = common::filesystem::join_path(LIB_AMATH_DIR, LIB_AMATH_NAME);

  if (common::filesystem::exists(lib_amath_redistributable_path)) {
    path = lib_path;
  } else if (common::filesystem::exists(lib_amath_path)) {
    path = replacePathPlaceholders(LIB_AMATH_DIR);
  }
#endif

  return path;
}

std::string getLibSvmlDir() {
  static std::string path;
  if (!path.empty())
    return path;

  const auto lib_path = common::filesystem::get_lib_directory();

#ifdef SVML_AVAILABLE
  std::string lib_svml_redistributable_path =
      common::filesystem::join_path(lib_path, LIB_SVML_NAME);
  std::string lib_svml_path = common::filesystem::join_path(LIB_SVML_DIR, LIB_SVML_NAME);

  std::string lib_intlc_redistributable_path =
      common::filesystem::join_path(lib_path, LIB_INTLC_NAME);
  std::string lib_intlc_path = common::filesystem::join_path(LIB_SVML_DIR, LIB_INTLC_NAME);

  if (common::filesystem::exists(lib_svml_redistributable_path) &&
      common::filesystem::exists(lib_intlc_redistributable_path)) {
    path = lib_path;
  } else if (common::filesystem::exists(lib_svml_path) &&
             common::filesystem::exists(lib_intlc_path)) {
    path = replacePathPlaceholders(LIB_SVML_DIR);
  }
#endif

  return path;
}

std::string getLibMvecDir() {
  static std::string path;
  if (!path.empty())
    return path;

  const auto lib_path = common::filesystem::get_lib_directory();

#ifdef LIBMVEC_AVAILABLE
  std::string lib_mvec_redistributable_path =
      common::filesystem::join_path(lib_path, LIB_MVEC_NAME);
  std::string lib_mvec_path = common::filesystem::join_path(LIB_MVEC_DIR, LIB_MVEC_NAME);

  if (common::filesystem::exists(lib_mvec_redistributable_path)) {
    path = lib_path;
  } else if (common::filesystem::exists(lib_mvec_path)) {
    path = replacePathPlaceholders(LIB_MVEC_DIR);
  }
#endif

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

int executeAndWait(
    llvm::StringRef Program,
    llvm::ArrayRef<llvm::StringRef> Args,
    std::optional<llvm::ArrayRef<llvm::StringRef>> Env,
    llvm::ArrayRef<std::optional<llvm::StringRef>> Redirects) {
#ifndef _WIN32
  return llvm::sys::ExecuteAndWait(Program, Args, Env, Redirects);
#else
  std::string ErrMsg;
  bool ExecutionFailed = false;

  llvm::SmallVector<std::optional<llvm::StringRef>, 3> ActualRedirects;
  llvm::SmallString<128> StdoutFile;
  llvm::SmallString<128> StderrFile;

  bool CaptureOutput = Redirects.empty();

  if(CaptureOutput) {
    if(auto E = llvm::sys::fs::createTemporaryFile(
           "acpp-tool-stdout", "txt", StdoutFile, llvm::sys::fs::OF_None))
      return -1;

    if(auto E = llvm::sys::fs::createTemporaryFile(
           "acpp-tool-stderr", "txt", StderrFile, llvm::sys::fs::OF_None)) {
      llvm::sys::fs::remove(StdoutFile);
      return -1;
    }

    ActualRedirects.push_back(llvm::StringRef{}); // stdin -> NUL
    ActualRedirects.push_back(StdoutFile.str());  // stdout -> temp file
    ActualRedirects.push_back(StderrFile.str());  // stderr -> temp file

    Redirects = ActualRedirects;
  }

  auto cleanup = [&]() {
    if(CaptureOutput) {
      auto Err0 = llvm::sys::fs::remove(StdoutFile);
      auto Err1 = llvm::sys::fs::remove(StderrFile);
    }
  };

  auto ProcessInfo =
      llvm::sys::ExecuteNoWait(Program, Args, Env, Redirects,
                               0, &ErrMsg, &ExecutionFailed, nullptr, true);

  if(ExecutionFailed) {
    cleanup();
    return -1;
  }

  auto Result = llvm::sys::Wait(ProcessInfo, std::nullopt);

  if(CaptureOutput) {
    if(auto StdoutBuffer = llvm::MemoryBuffer::getFile(StdoutFile))
      llvm::outs() << StdoutBuffer.get()->getBuffer();

    if(auto StderrBuffer = llvm::MemoryBuffer::getFile(StderrFile))
      llvm::errs() << StderrBuffer.get()->getBuffer();
  }

  cleanup();
  return Result.ReturnCode;
#endif
}

} // namespace compiler
} // namespace hipsycl
