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

#include "hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtx.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DebugInfo.h>
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



class LibdevicePath {
public:
  static bool get(std::string& Out) {
    static LibdevicePath P;

    if(P.IsFound)
      Out = P.Path;
    return P.IsFound;
  }
private:
  LibdevicePath() {
    IsFound = findLibdevice(Path);

    if(IsFound) {
      HIPSYCL_DEBUG_INFO << "LLVMToPtx: Found libdevice: " << Path << "\n";
    } else {
      HIPSYCL_DEBUG_INFO << "LLVMToPtx: Could not find CUDA libdevice!\n";
    }
  }

  bool findLibdevice(std::string& Out) {
    
    std::string CUDAPath = HIPSYCL_CUDA_PATH;
    std::vector<std::string> SubDir {"nvvm", "libdevice"};
    std::string BitcodeDir = common::filesystem::join_path(CUDAPath, SubDir);

    try {
      auto Files = common::filesystem::list_regular_files(BitcodeDir);
      for(const auto& F : Files) {
        if (F.find("libdevice.") != std::string::npos && F.find(".bc") != std::string::npos) {
          Out = F;
          return true;
        }
      }
    }catch(...) { /* false will be returned anyway at this point */ }

    return false;
  }

  std::string Path;
  bool IsFound;
};

}

LLVMToPtxTranslator::LLVMToPtxTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::sscp::backend::ptx, KN}, KernelNames{KN} {}


bool LLVMToPtxTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  std::string Triple = "nvptx64-nvidia-cuda";
  std::string DataLayout =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-"
      "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

  M.setTargetTriple(Triple);
  M.setDataLayout(DataLayout);

  AddressSpaceMap ASMap = getAddressSpaceMap();
  
  KernelFunctionParameterRewriter ParamRewriter{
      // PTX wants ByVal attribute for all aggregates passed in by-value
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
      // Those pointers to by-value data can be in generic AS
      ASMap[AddressSpace::Generic],
      // Actual pointers should be in global memory
      ASMap[AddressSpace::Global]};
  
  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      
      llvm::SmallVector<llvm::Metadata*, 4> Operands;
      Operands.push_back(llvm::ValueAsMetadata::get(F));
      Operands.push_back(llvm::MDString::get(M.getContext(), "kernel"));
      Operands.push_back(llvm::ValueAsMetadata::getConstant(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 1)));

      M.getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDTuple::get(M.getContext(), Operands));

      F->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    }
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-ptx-full.bc"});
  
  std::string LibdeviceFile;
  if(!LibdevicePath::get(LibdeviceFile)) {
    this->registerError("LLVMToPtx: Could not find CUDA libdevice bitcode library");
    return false;
  }

  AddressSpaceInferencePass ASIPass {ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  // It seems there is an issue with debug info in PTX, so strip it for now
  // TODO: We should attempt to find out what exactly is causing the problem
  // so that code still can be debugged on NVIDIA GPUs.
  llvm::StripDebugInfo(M);

  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;
  if(!this->linkBitcodeFile(M, LibdeviceFile, Triple, DataLayout))
    return false;

  return true;
}

bool LLVMToPtxTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-ptx-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-ptx-%%%%%%.s");
  
  std::string OutputFilename = OutputFile->TmpName;
  
  auto E = InputFile.takeError();
  if(E){
    this->registerError("LLVMToPtx: Could not create temp file: "+InputFile->TmpName);
    return false;
  }

  AtScopeExit DestroyInputFile([&]() { auto Err = InputFile->discard(); });
  AtScopeExit DestroyOutputFile([&]() { auto Err = OutputFile->discard(); });

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};
  
  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();

  std::string ClangPath = HIPSYCL_CLANG_PATH;

  std::string PtxVersionArg = "+ptx" + std::to_string(PtxVersion);
  std::string PtxTargetArg = "sm_" + std::to_string(PtxTarget);
  llvm::SmallVector<llvm::StringRef, 16> Invocation{ClangPath,
                                                    "-cc1",
                                                    "-triple",
                                                    "nvptx64-nvidia-cuda",
                                                    "-target-feature",
                                                    PtxVersionArg,
                                                    "-target-cpu",
                                                    PtxTargetArg,
                                                    "-O3",
                                                    "-S",
                                                    "-x",
                                                    "ir",
                                                    "-o",
                                                    OutputFilename,
                                                    InputFile->TmpName};

  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToPtx: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      ClangPath, Invocation);
  
  if(R != 0) {
    this->registerError("LLVMToPtx: clang invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToPtx: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToPtxTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "ptx-version") {
    this->PtxVersion = std::stoi(Value);
    return true;
  } else if(Option == "ptx-target-device") {
    this->PtxTarget = std::stoi(Value);
    return true;
  }

  return false;
}

bool LLVMToPtxTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  for(const auto& Name : KernelNames)
    if(F.getName() == Name)
      return true;
  return false;
}

AddressSpaceMap LLVMToPtxTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 5;
  ASMap[AddressSpace::Constant] = 4;
  // NVVM wants to have allocas in address space 0
  ASMap[AddressSpace::AllocaDefault] = 0;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 4;

  return ASMap;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToPtxTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToPtxTranslator>(KernelNames);
}

}
}
