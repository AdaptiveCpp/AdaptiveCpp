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
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
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
#include <string>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

namespace {

static const char* DynamicLocalMemArrayName = "__hipsycl_sscp_spirv_dynamic_local_mem";

bool setDynamicLocalMemoryCapacity(llvm::Module& M, unsigned numBytes) {
  llvm::GlobalVariable* GV = M.getGlobalVariable(DynamicLocalMemArrayName);

  if(!GV) {
    // If non-zero number of bytes are needed, not finding the global variable is
    // an error.
    return numBytes == 0;
  }

  if(numBytes > 0) {
    unsigned AddressSpace = GV->getAddressSpace();
    unsigned numInts = (numBytes + 4 - 1) / 4;
    llvm::Type* T = llvm::ArrayType::get(llvm::Type::getInt32Ty(M.getContext()), numInts);

    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(T),
        GV->getName() + ".resized", nullptr, llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal,
        AddressSpace);

    NewVar->setAlignment(GV->getAlign());
    llvm::Value* V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());
    GV->replaceAllUsesWith(V);
    GV->eraseFromParent();
  }
  return true;
}

bool removeDynamicLocalMemorySupport(llvm::Module& M) {
  llvm::GlobalVariable* GV = M.getGlobalVariable(DynamicLocalMemArrayName);
  if(GV) {
    GV->replaceAllUsesWith(llvm::ConstantPointerNull::get(GV->getType()));
    GV->eraseFromParent();
  }
  return true;
}

}

LLVMToSpirvTranslator::LLVMToSpirvTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::sscp::backend::spirv, KN}, KernelNames{KN} {}


bool LLVMToSpirvTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  
  M.setTargetTriple("spir64-unknown-unknown");
  M.setDataLayout(
      "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024");

  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      // SPIR-V translator wants to have structures like our kernel lambda
      // be passed in as pointers with ByVal attribute
      forceAllUsedPointerArgumentsToByVal(M, F);
    }
  }

  AddressSpaceMap ASMap;

  // By default, llvm-spirv translator uses the mapping where
  // ASMap[AddressSpace::Generic] = 4;
  // ASMap[AddressSpace::Private] = 0;
  // We currently require a different mapping where the default address
  // space is the generic address space, which requires a patched llvm-spirv.
  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 4;
  ASMap[AddressSpace::Constant] = 2;
  ASMap[AddressSpace::AllocaDefault] = 4;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;

  // llvm-spirv translator expects by-value kernel arguments such as our
  // kernel lambda to be passed in through private address space
  rewriteKernelArgumentAddressSpacesTo(ASMap[AddressSpace::Private], M, KernelNames, PH);

  for(auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Setting up kernel " << KernelName << "\n";
    if(auto* F = M.getFunction(KernelName)) {
      F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    }
  }

  for(auto& F : M.getFunctionList()) {
    if(F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL){
      // All functions must be marked as spir_func
      if(F.getCallingConv() != llvm::CallingConv::SPIR_FUNC)
        F.setCallingConv(llvm::CallingConv::SPIR_FUNC);
      
      // All callers must use spir_func calling convention
      for(auto U : F.users()) {
        if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
          CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
        }
      }
    }
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-spirv-full.bc"});
  
  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  // Set up local memory
  if(DynamicLocalMemSize > 0) {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Configuring kernel for " << DynamicLocalMemSize
                       << " bytes of local memory\n";
    if(!setDynamicLocalMemoryCapacity(M, DynamicLocalMemSize)) {
      this->registerError("Could not set dynamic local memory size");
      return false;
    }
  } else {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Removing dynamic local memory support from module\n";
    removeDynamicLocalMemorySupport(M);
  }

  AddressSpaceInferencePass ASIPass{ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  return true;
}


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

  std::string OutputArg = "-o=" + OutputFilename;
  llvm::SmallVector<llvm::StringRef, 16> Invocation{LLVMSpirVTranslator, OutputArg,
                                                    InputFile->TmpName};
  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      LLVMSpirVTranslator, Invocation);

  if(R != 0) {
    this->registerError("LLVMToSpirv: llvm-spirv invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToSpirv: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToSpirvTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "spirv-dynamic-local-mem-allocation-size") {
    this->DynamicLocalMemSize = static_cast<unsigned>(std::stoi(Value));
    return true;
  }

  return false;
}

bool LLVMToSpirvTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToSpirvTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToSpirvTranslator>(KernelNames);
}


}
}
