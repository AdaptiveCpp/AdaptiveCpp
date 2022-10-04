/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#ifndef HIPSYCL_LLVM_TO_BACKEND_HPP
#define HIPSYCL_LLVM_TO_BACKEND_HPP



#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <typeinfo>
#include <functional>
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace llvm {
class Module;
}

namespace hipsycl {
namespace compiler {

struct PassHandler;

struct TranslationHints {
  std::optional<std::size_t> RequestedLocalMemSize;
  std::optional<std::size_t> SubgroupSize;
  std::optional<rt::range<3>> WorkGroupSize;
};

class LLVMToBackendTranslator {
public:
  LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
    const std::vector<std::string>& OutliningEntrypoints);

  virtual ~LLVMToBackendTranslator() {}

  template<auto& ConstantName, class T>
  void setS2IRConstant(const T& value) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "Unsupported type for S2 IR constant");

    std::string name = typeid(__hipsycl_sscp_s2_ir_constant<ConstantName, T>).name();
    setS2IRConstant<T>(name, value);
  }

  virtual bool setBuildOptions(const std::string& Opts) {
    return true;
  }
  
  bool fullTransformation(const std::string& LLVMIR, std::string& out);
  bool prepareIR(llvm::Module& M);
  bool translatePreparedIR(llvm::Module& FlavoredModule, std::string& out);


  const std::vector<std::string>& getErrorLog() const {
    return Errors;
  }

protected:
  bool linkBitcodeFile(llvm::Module& M, const std::string& BitcodeFile);
  bool linkBitcodeString(llvm::Module& M, const std::string& Bitcode);
  // If backend needs to set IR constants, it should do so here.
  virtual bool prepareBackendFlavor(llvm::Module& M) = 0;
  // Transform LLVM IR as much as required to backend-specific flavor
  virtual bool toBackendFlavor(llvm::Module &M) = 0;
  virtual bool translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) = 0;

  // By default, just runs regular O3 pipeline. Backends may override
  // if they want to do something more specific.
  virtual bool optimizeFlavoredIR(llvm::Module& M, PassHandler& PH);

  void registerError(const std::string& E) {
    Errors.push_back(E);
  }
private:
  template<class T>
  void setS2IRConstant(const std::string& name, T value);

  int S2IRConstantBackendId;
  std::vector<std::string> OutliningEntrypoints;
  std::vector<std::string> Errors;
  std::unordered_map<std::string, std::function<void(llvm::Module &)>> S2IRConstantApplicators;
};

}
}

#endif
