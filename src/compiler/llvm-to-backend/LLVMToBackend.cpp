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

#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include <cstdint>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <string>

namespace hipsycl {
namespace compiler {


LLVMToBackendTranslator::LLVMToBackendTranslator(int S2IRConstantCurrentBackendId)
: S2IRConstantBackendId(S2IRConstantCurrentBackendId) {
  setS2IRConstant<sycl::sscp::current_backend, int>(
      S2IRConstantCurrentBackendId);
}

template<class T>
void LLVMToBackendTranslator::setS2IRConstant(const std::string& name, T value){
  S2IRConstantApplicators[name] = [=](llvm::Module& M){
    S2IRConstant C = S2IRConstant::getFromConstantName(M, name);
    C.set<T>(value);
  };
}

bool LLVMToBackendTranslator::fullTransformation(const std::string &LLVMIR, std::string &out) {
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> M;
  auto err = loadModuleFromString(LLVMIR, ctx, M);

  if (!err.success())
    return false;

  assert(M);
  if (!prepareIR(*M))
    return false;
  if (!translatePreparedIR(*M, out))
    return false;

  return true;
}

bool LLVMToBackendTranslator::prepareIR(llvm::Module &M) {
  if(!this->prepareBackendFlavor(M))
    return false;
  
  for(auto& A : S2IRConstantApplicators) {
    A.second(M);
  }

  bool ContainsUnsetIRConstants = false;
  bool FlavoringResult = false;

  constructPassBuilderAndMAM([&](llvm::PassBuilder &PB, llvm::ModuleAnalysisManager &MAM) {
    // Optimize away unnecessary branches due to backend-specific S2IR constants
    // This is what allows us to specialize code for different backends.
    IRConstant::optimizeCodeAfterConstantModification(M, MAM);
    // Rerun kernel outlining pass so that we don't include unneeded functions
    // that are specific to other backends.
    KernelOutliningPass KP;
    KP.run(M, MAM);

    FlavoringResult = this->toBackendFlavor(M);
    if(FlavoringResult) {
      // Run optimizations
      llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
      MPM.run(M, MAM);

      S2IRConstant::forEachS2IRConstant(M, [&](S2IRConstant C) {
        if (C.isValid()) {
          if (!C.isInitialized()) {
            ContainsUnsetIRConstants = true;
            this->registerError("hipSYCL S2IR constant was not set: " +
                                C.getGlobalVariable()->getName().str());
          }
        }
      });
    }
  });

  return FlavoringResult && !ContainsUnsetIRConstants;
}

bool LLVMToBackendTranslator::translatePreparedIR(llvm::Module &FlavoredModule, std::string &out) {
  return this->translateToBackendFormat(FlavoredModule, out);
}

#define HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(type)                                              \
  template void LLVMToBackendTranslator::setS2IRConstant<type>(const std::string &, type);

HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(int8_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(uint8_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(int16_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(uint16_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(int32_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(uint32_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(int64_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(uint64_t)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(float)
HIPSYCL_INSTANTIATE_S2IRCONSTANT_SETTER(double)

}
}

