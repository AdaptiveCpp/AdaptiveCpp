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
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>
#include <memory>
#include <cassert>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

LLVMToSpirvTranslator::LLVMToSpirvTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::sscp::backend::spirv, KN}, KernelNames{KN} {}


bool LLVMToSpirvTranslator::toBackendFlavor(llvm::Module &M) {
  M.setTargetTriple("spir64-unknown-unknown");
  M.setDataLayout(
      "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024");

  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    }
  }

  for(auto& F : M.getFunctionList()) {
    if(F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL &&
      F.getCallingConv() != llvm::CallingConv::SPIR_FUNC)
      F.setCallingConv(llvm::CallingConv::SPIR_FUNC);
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-spirv-full.bc"});
  
  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  return true;
}


template<class F>
class AtScopeExit {
public:
  AtScopeExit(F&& f)
  : Handler(f) {}

  ~AtScopeExit() {Handler();}
private:
  std::function<void()> Handler;
};

bool LLVMToSpirvTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-spirv-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-spirv-%%%%%%.spv");
  
  std::string OutputFilename = OutputFile->TmpName;
  
  auto E = InputFile.takeError();
  if(E){
    this->registerError("LLVMToSpirv: Could not create temp file: "+InputFile->TmpName);
    return false;
  }

  AtScopeExit DestroyInputFile([&]() { auto Err = InputFile->discard(); });
  AtScopeExit DestroyOutputFile([&]() { auto Err = OutputFile->discard(); });

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};
  
  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();

  std::string LLVMSpirVTranslator = hipsycl::common::filesystem::join_path(
      hipsycl::common::filesystem::get_install_directory(), HIPSYCL_RELATIVE_LLVMSPIRV_PATH);

  HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Invoking " << LLVMSpirVTranslator << "\n";

  int R = llvm::sys::ExecuteAndWait(
      LLVMSpirVTranslator, {LLVMSpirVTranslator, "-o=" + OutputFilename, InputFile->TmpName});
  if(R != 0) {
    this->registerError("LLVMToSpirv: llvm-spirv invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getOpenFile(OutputFile->FD, OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToSpirv: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();

  return true;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToSpirvTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToSpirvTranslator>(KernelNames);
}

}
}
