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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"

#include <cstdint>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <string>

namespace hipsycl {
namespace compiler {


LLVMToBackendTranslator::LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
  const std::vector<std::string>& OutliningEPs)
: S2IRConstantBackendId(S2IRConstantCurrentBackendId), OutliningEntrypoints{OutliningEPs} {
  setS2IRConstant<sycl::sscp::current_backend, int>(
      S2IRConstantCurrentBackendId);
}

bool LLVMToBackendTranslator::partialTransformation(const std::string &LLVMIR, std::string &Out) {
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> M;
  auto err = loadModuleFromString(LLVMIR, ctx, M);

  if (err) {
    this->registerError("LLVMToBackend: Could not load LLVM module");
    llvm::handleAllErrors(std::move(err), [&](llvm::ErrorInfoBase &EIB) {
      this->registerError(EIB.message());
    });
    return false;
  }

  assert(M);
  if (!prepareIR(*M))
    return false;
  
  llvm::raw_string_ostream OutputStream{Out};
  llvm::WriteBitcodeToFile(*M, OutputStream);

  return true;
}

bool LLVMToBackendTranslator::fullTransformation(const std::string &LLVMIR, std::string &out) {
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> M;
  auto err = loadModuleFromString(LLVMIR, ctx, M);

  if (err) {
    this->registerError("LLVMToBackend: Could not load LLVM module");
    llvm::handleAllErrors(std::move(err), [&](llvm::ErrorInfoBase &EIB) {
      this->registerError(EIB.message());
    });
    return false;
  }

  assert(M);
  if (!prepareIR(*M))
    return false;
  if (!translatePreparedIR(*M, out))
    return false;

  return true;
}

bool LLVMToBackendTranslator::prepareIR(llvm::Module &M) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Preparing backend flavoring...\n";

  if(!this->prepareBackendFlavor(M))
    return false;
  
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Applying S2 IR constants...\n";
  for(auto& A : S2IRConstantApplicators) {
    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Setting S2 IR constant " << A.first << "\n";
    A.second(M);
  }

  bool ContainsUnsetIRConstants = false;
  bool FlavoringSuccessful = false;
  bool OptimizationSuccessful = false;

  constructPassBuilderAndMAM([&](llvm::PassBuilder &PB, llvm::ModuleAnalysisManager &MAM) {
    // Optimize away unnecessary branches due to backend-specific S2IR constants
    // This is what allows us to specialize code for different backends.
    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Optimizing branches post S2 IR constant application...\n";
    IRConstant::optimizeCodeAfterConstantModification(M, MAM);
    // Rerun kernel outlining pass so that we don't include unneeded functions
    // that are specific to other backends.
    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Reoutlining kernels...\n";
    KernelOutliningPass KP{OutliningEntrypoints};
    KP.run(M, MAM);

    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Adding backend-specific flavor to IR...\n";
    FlavoringSuccessful = this->toBackendFlavor(M);
    if(FlavoringSuccessful) {
      // Run optimizations
      HIPSYCL_DEBUG_INFO << "LLVMToBackend: Optimizing flavored IR...\n";
      PassHandler PH {&PB, &MAM};
      OptimizationSuccessful = optimizeFlavoredIR(M, PH);
      if(!OptimizationSuccessful) {
        this->registerError("LLVMToBackend: Optimization failed");
      }

      S2IRConstant::forEachS2IRConstant(M, [&](S2IRConstant C) {
        if (C.isValid()) {
          if (!C.isInitialized()) {
            ContainsUnsetIRConstants = true;
            this->registerError("LLVMToBackend: hipSYCL S2IR constant was not set: " +
                                C.getGlobalVariable()->getName().str());
          }
        }
      });
    } else {
      HIPSYCL_DEBUG_INFO << "LLVMToBackend: Flavoring failed\n";
    }
  });

  return FlavoringSuccessful && OptimizationSuccessful && !ContainsUnsetIRConstants;
}

bool LLVMToBackendTranslator::translatePreparedIR(llvm::Module &FlavoredModule, std::string &out) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Invoking translation to backend-specific format\n";
  return this->translateToBackendFormat(FlavoredModule, out);
}

bool LLVMToBackendTranslator::optimizeFlavoredIR(llvm::Module& M, PassHandler& PH) {
  assert(PH.PassBuilder);
  assert(PH.ModuleAnalysisManager);

  llvm::ModulePassManager MPM =
      PH.PassBuilder->buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  MPM.run(M, *PH.ModuleAnalysisManager);
  return true;
}

bool LLVMToBackendTranslator::linkBitcodeString(llvm::Module &M, const std::string &Bitcode) {
  std::unique_ptr<llvm::Module> OtherModule;
  auto err = loadModuleFromString(Bitcode, M.getContext(), OtherModule);

  if (err) {
    this->registerError("LLVMToBackend: Could not load LLVM module");
    llvm::handleAllErrors(std::move(err), [&](llvm::ErrorInfoBase &EIB) {
      this->registerError(EIB.message());
    });
    return false;
  }

  // Returns true on error
  if (llvm::Linker::linkModules(M, std::move(OtherModule),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    this->registerError("LLVMToBackend: Linking module failed");
    return false;
  }
  return true;
}

bool LLVMToBackendTranslator::linkBitcodeFile(llvm::Module& M, const std::string& BitcodeFile) {
  auto F = llvm::MemoryBuffer::getFile(BitcodeFile);
  if(auto Err = F.getError()) {
    this->registerError("LLVMToBackend: Could not open file " + BitcodeFile);
    return false;
  }
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Linking with bitcode file: " << BitcodeFile << "\n";
  return linkBitcodeString(M, std::string{F.get()->getBuffer()});
}

void LLVMToBackendTranslator::setS2IRConstant(const std::string &name, const void *ValueBuffer) {
  S2IRConstantApplicators[name] = [=](llvm::Module& M){
    S2IRConstant C = S2IRConstant::getFromConstantName(M, name);
    C.set(ValueBuffer);
  };
}
}
}

