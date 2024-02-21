/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay
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

#include "hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/KernelFlattening.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"
#include "hipSYCL/compiler/cbs/SimplifyKernel.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/HostKernelWrapperPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR < 16
#include <llvm/Support/Host.h>
#include <llvm/ADT/Triple.h>
#else
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#endif

#include <cassert>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

LLVMToHostTranslator::LLVMToHostTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::sscp::backend::host, KN}, KernelNames{KN} {}

bool LLVMToHostTranslator::toBackendFlavor(llvm::Module &M, PassHandler &PH) {

  {
    std::error_code EC;
    llvm::raw_fd_ostream rs{"hipsycl-sscp-host-before.ll", EC};
    M.print(rs, nullptr);
  }

  std::string Triple = llvm::sys::getProcessTriple();
  // Fixme:
  // std::string DataLayout =
  // "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";

  M.setTargetTriple(Triple);
  // M.setDataLayout(DataLayout);

  AddressSpaceMap ASMap = getAddressSpaceMap();

  for (auto KernelName : KernelNames) {
    if (auto *F = M.getFunction(KernelName)) {

      llvm::SmallVector<llvm::Metadata *, 4> Operands;
      Operands.push_back(llvm::ValueAsMetadata::get(F));
      Operands.push_back(llvm::MDString::get(M.getContext(), "kernel"));
      Operands.push_back(llvm::ValueAsMetadata::getConstant(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 1)));

      M.getOrInsertNamedMetadata(SscpAnnotationsName)
          ->addOperand(llvm::MDTuple::get(M.getContext(), Operands));

      F->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    }
  }

  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(common::filesystem::get_install_directory(),
                                    {"lib", "hipSYCL", "bitcode", "libkernel-sscp-host-full.bc"});

  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  llvm::ModulePassManager MPM;

  MPM.addPass(AddressSpaceInferencePass{ASMap});

  PH.PassBuilder->registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager &MAM) {
    MAM.registerPass([] { return SplitterAnnotationAnalysis{}; });
  });
  PH.PassBuilder->registerModuleAnalyses(*PH.ModuleAnalysisManager);
  registerCBSPipeline(MPM, hipsycl::compiler::OptLevel::O3, true);

  MPM.run(M, *PH.ModuleAnalysisManager);

  {
    std::error_code EC;
    llvm::raw_fd_ostream rs{"hipsycl-sscp-host-cbs.ll", EC};
    M.print(rs, nullptr);
  }

  return true;
}

bool LLVMToHostTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {
  {
    std::error_code EC;
    llvm::raw_fd_ostream rs{"hipsycl-sscp-host.ll", EC};
    FlavoredModule.print(rs, nullptr);
  }

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-host-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-host-%%%%%%.so");

  if (auto E = InputFile.takeError()) {
    this->registerError("LLVMToHost: Could not create temp file: " + InputFile->TmpName);
    return false;
  }

  if (auto E = OutputFile.takeError()) {
    this->registerError("LLVMToHost: Could not create temp file: " + OutputFile->TmpName);
    return false;
  }

  std::string OutputFilename = OutputFile->TmpName;

  AtScopeExit DestroyInputFile([&]() {
    if (InputFile->discard())
      ;
  });
  AtScopeExit DestroyOutputFile([&]() {
    if (OutputFile->discard())
      ;
  });

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};

  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();

  std::string ClangPath = HIPSYCL_CLANG_PATH;

  llvm::SmallVector<llvm::StringRef, 16> Invocation{
      ClangPath, "-O3", "-march=native", "-x", "ir", "-shared", "-o", OutputFilename, InputFile->TmpName};

  std::string ArgString;
  for (const auto &S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToHost: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(ClangPath, Invocation);

  if (R != 0) {
    this->registerError("LLVMToHost: clang invocation failed with exit code " + std::to_string(R));
    return false;
  }

  auto ReadResult = llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);

  if (auto Err = ReadResult.getError()) {
    this->registerError("LLVMToHost: Could not read result file" + Err.message());
    return false;
  }

  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToHostTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  return false;
}

bool LLVMToHostTranslator::isKernelAfterFlavoring(llvm::Function &F) {
  for (const auto &Name : KernelNames)
    if (F.getName() == Name)
      return true;
  return false;
}

AddressSpaceMap LLVMToHostTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;
  // TODO for CPU
  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 0;
  ASMap[AddressSpace::Local] = 0;
  ASMap[AddressSpace::Private] = 0;
  ASMap[AddressSpace::Constant] = 0;
  // NVVM wants to have allocas in address space 0
  ASMap[AddressSpace::AllocaDefault] = 0;
  ASMap[AddressSpace::GlobalVariableDefault] = 0;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 0;

  return ASMap;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToHostTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToHostTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl
