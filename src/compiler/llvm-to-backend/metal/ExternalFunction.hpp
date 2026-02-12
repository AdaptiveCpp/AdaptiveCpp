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
#ifndef HIPSYCL_COMPILER_EXTERNALFUNCTION_HPP
#define HIPSYCL_COMPILER_EXTERNALFUNCTION_HPP

#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace llvm {
class CallInst;
class Value;
class Type;
} // namespace llvm

namespace hipsycl {
namespace compiler {

struct ExternalFunctionInfo {
  std::string name;
  std::optional<std::string> replacement;
  std::optional<std::string> code;
  bool exactMatch = true;
  bool used = false;
  std::vector<std::string> deps;
  bool convertToVar = false;
  bool ignore = false;
  int argsCount = -1;
  bool needsLocalMemory = false;
  std::function<std::optional<std::string>(const llvm::CallInst*, std::string& errorStr)> customCallEmitter = nullptr;
};

class ExternalFunctionMapper {
public:
  ExternalFunctionMapper(std::function<std::string(const llvm::Value*)> addrSpaceMapper = nullptr, std::function<std::string(const llvm::Value*)> exprMapper = nullptr, std::function<std::string(const llvm::Type*)> typeMapper = nullptr);

  const ExternalFunctionInfo* getFunctionInfo(std::string_view name);
  std::vector<const ExternalFunctionInfo*> getUsedFunctions();

private:
  std::vector<ExternalFunctionInfo> externalFunctions;
  std::function<std::string(const llvm::Value*)> addrSpaceMapper;
  std::function<std::string(const llvm::Value*)> exprMapper;
  std::function<std::string(const llvm::Type*)> typeMapper;

  using PrefixMap = std::map<std::string_view, ExternalFunctionInfo*>;
  PrefixMap map;

  void initializeMap();
  std::optional<std::string> emitMetalInlineCall(const llvm::CallInst* CI, std::string& errorStr);
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_COMPILER_EXTERNALFUNCTION_HPP