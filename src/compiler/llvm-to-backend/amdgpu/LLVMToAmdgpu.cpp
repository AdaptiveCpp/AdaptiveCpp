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

#include "hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/DerivedTypes.h>
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

class RocmDeviceLibs {
public:
  static bool getOclAbiVersionLibrary(const std::string &DeviceLibsPath,
                                      std::string &OclAbiVersionLibOut) {
    static std::string OclABIVersionLib;
    int MaxABIVersion = 0;

    if(OclABIVersionLib.empty()) {
      auto Files = common::filesystem::list_regular_files(DeviceLibsPath);
      for(const auto& F : Files) {
        std::string Begin = "oclc_abi_version_";
        auto PosBeg = F.find(Begin);
        auto PosEnd = F.find(".bc");
        if (PosBeg != std::string::npos && PosEnd != std::string::npos) {
          std::string ABIVersionString =
              F.substr(PosBeg + Begin.size(), PosEnd - PosBeg - Begin.size());
          int ABIVersion = std::stoi(ABIVersionString);
          if(ABIVersion > MaxABIVersion) {
            OclABIVersionLib = F;
            MaxABIVersion = ABIVersion;
          }
        }
      }
    }

    OclAbiVersionLibOut = OclABIVersionLib;
    return !OclABIVersionLib.empty();
  }
};

LLVMToAmdgpuTranslator::LLVMToAmdgpuTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::sscp::backend::amdgpu, KN}, KernelNames{KN} {
  RocmDeviceLibsPath = common::filesystem::join_path(HIPSYCL_ROCM_PATH,
                                                     std::vector<std::string>{"amdgcn", "bitcode"});
}


bool LLVMToAmdgpuTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  
  M.setTargetTriple("amdgcn-amd-amdhsa");
  M.setDataLayout(
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-"
      "v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");


  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      // AMDGPU backend expects arguments to be passed as byref instead of byval
      for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
        if(F->hasParamAttribute(i, llvm::Attribute::ByVal)) {
          auto ByValAttr = F->getParamAttribute(i, llvm::Attribute::ByVal);
          llvm::Type* ParamPointeeType = ByValAttr.getValueAsType();
          F->removeParamAttr(i, llvm::Attribute::ByVal);
          
          if(!F->hasParamAttribute(i, llvm::Attribute::ByRef)) {
            F->addParamAttr(i, llvm::Attribute::getWithByRefType(M.getContext(), ParamPointeeType));
          }
        }
      }
    }
  }

  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 4;
  ASMap[AddressSpace::Constant] = 2;

  rewriteKernelArgumentAddressSpacesTo(ASMap[AddressSpace::Private], M, KernelNames, PH);
  
  for(auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Setting up kernel " << KernelName << "\n";
    if(auto* F = M.getFunction(KernelName)) {
      F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    }
  }
  
  for(auto& F : M.getFunctionList()) {
    if(F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
      // When we are already lowering to device specific format,
      // we can expect that we have no external users anymore.
      // All linking should be done by now. The exception are intrinsics.
      if(!F.isIntrinsic() && F.getName().find("__hipsycl_sscp") == std::string::npos)
        F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  // TODO handle address spaces

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-amdgpu-amdhsa-full.bc"});
  
  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  // Needed as a workaround for some ROCm versions
#ifdef HIPSYCL_SSCP_AMDGPU_FORCE_OCLC_ABI_VERSION
  std::string OclABIVersionLib;
  if (!RocmDeviceLibs::getOclAbiVersionLibrary(RocmDeviceLibsPath, OclABIVersionLib)) {
    this->registerError("Could not find ROCm oclc ABI version bitcode library");
    return false;
  }
  if(!this->linkBitcodeFile(M, OclABIVersionLib))
    return false;
#endif
  return true;
}


bool LLVMToAmdgpuTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-amdgpu-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-amdgpu-%%%%%%.hipfb");
  
  std::string OutputFilename = OutputFile->TmpName;
  
  auto E = InputFile.takeError();
  if(E){
    this->registerError("LLVMToAmdgpu: Could not create temp file: "+InputFile->TmpName);
    return false;
  }

  AtScopeExit DestroyInputFile([&]() { auto Err = InputFile->discard(); });
  AtScopeExit DestroyOutputFile([&]() { auto Err = OutputFile->discard(); });

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};
  
  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();

  llvm::SmallVector<llvm::StringRef, 16> Invocation;
  auto OffloadArchFlag = "--offload-arch="+TargetDevice;
  auto RocmPathFlag = "--rocm-path="+std::string{HIPSYCL_ROCM_PATH};
  auto RocmDeviceLibsFlag = "--rocm-device-lib-path="+RocmDeviceLibsPath;
  
  std::string ClangPath = HIPSYCL_CLANG_PATH;

  if(OnlyGenerateAssembly) {
    Invocation = {ClangPath, "-cc1",
                  "-triple", "amdgcn-amd-amdhsa",
                  "-target-cpu", TargetDevice,
                  "-O3",
                  "-S",
                  "-x", "ir",
                  "-mllvm", "-amdgpu-early-inline-all=true",
                  "-mllvm", "-amdgpu-function-calls=false",
                  "-o",
                  OutputFilename,
                  InputFile->TmpName};
  } else {
    
    Invocation = {ClangPath, "-x", "hip",
      "--cuda-device-only", "-O3",
      RocmPathFlag,
      RocmDeviceLibsFlag,
      OffloadArchFlag,
      "-mllvm", "-amdgpu-early-inline-all=true",
      "-mllvm", "-amdgpu-function-calls=false",
      "-Xclang", "-mlink-bitcode-file", "-Xclang", InputFile->TmpName,
      "-o", OutputFilename, 
      "/dev/null"};
  }

  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      ClangPath, Invocation);
  
  if(R != 0) {
    this->registerError("LLVMToAmdgpu: clang invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToAmdgpu: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();
  
  return true;
}

bool LLVMToAmdgpuTranslator::setBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "amdgpu-target-device") {
    TargetDevice = Value;
    return true;
  } else if (Option == "rocm-device-libs-path") {
    RocmDeviceLibsPath = Value;
    return true; 
  }

  return false;
}

bool LLVMToAmdgpuTranslator::setBuildFlag(const std::string &Flag) {
  if(Flag == "assemble-only") {
    OnlyGenerateAssembly = true;
    return true;
  }

  return false;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToAmdgpuTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToAmdgpuTranslator>(KernelNames);
}

}
}
