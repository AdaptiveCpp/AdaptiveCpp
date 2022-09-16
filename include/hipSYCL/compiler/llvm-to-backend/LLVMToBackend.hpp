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



#include <string>
#include <vector>
#include <typeinfo>
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"

namespace llvm {
class Module;
}

namespace hipsycl {
namespace compiler {

class IRConstantHandler {

};

class LLVMToBackendTranslator {
public:
  LLVMToBackendTranslator(int S2IRConstantCurrentBackendId);

  virtual ~LLVMToBackendTranslator() {}

  template<auto& ConstantName, class T>
  void setS2IRConstant(const T& value) {
    std::string name = typeid(__hipsycl_sscp_s2_ir_constant<ConstantName, T>).name();
  }

  virtual bool setBuildOptions(const std::string& Opts) {
    return true;
  }
  virtual bool fullTransformation(const std::string& LLVMIR, std::string& out) = 0;

  virtual bool toBackendFlavor(llvm::Module &M) = 0;
  virtual bool translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) = 0;

  const std::vector<std::string>& getErrorLog() const {
    return Errors;
  }

protected:
  void registerError(const std::string& E) {
    Errors.push_back(E);
  }
private:
  int S2IRConstantBackendId;
  std::vector<std::string> Errors;
  //std::vector<std::pair<std::string, std::
};

}
}

#endif
