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

#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirv.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <cassert>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

LLVMToSpirvTranslator::LLVMToSpirvTranslator(const std::vector<std::string> &KN)
    : KernelNames{KN} {}

bool LLVMToSpirvTranslator::fullTransformation(const std::string &LLVMIR, std::string &out) {
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> M;
  auto err = loadModuleFromString(LLVMIR, ctx, M);

  if(!err.success())
    return false;
  
  assert(M);
  if(!toBackendFlavor(*M))
    return false;
  if(!translateToBackendFormat(*M, out));
    return false;
  
  return true;
}

bool LLVMToSpirvTranslator::toBackendFlavor(llvm::Module &M) {
  M.setTargetTriple("spir64-unknown-unknown");
  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    }
  }
  return true;
}

bool LLVMToSpirvTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {
  constructPassBuilder([&](llvm::PassBuilder &PB, auto &LAM, auto &FAM, auto &CGAM, auto &MAM) {
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(FlavoredModule, MAM);
  });

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-spirv-%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-spirv-%%%%.spv");

  auto E = InputFile.takeError();
  if(!E.success()){
    this->registerError("Could not create temp file: "+InputFile->TmpName);
    return false;
  }

  E = OutputFile.takeError();
  if(!E.success()){
    this->registerError("Could not create temp file: "+OutputFile->TmpName);
    return false;
  }

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->TmpName, EC};
  
  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);

  if(!InputFile->discard().success() || !OutputFile->discard().success()) {
    this->registerError("Discarding temp file failed");
  }

  return true;
}

}
}
