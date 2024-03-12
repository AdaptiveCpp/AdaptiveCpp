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
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>
#include <algorithm>
#include <memory>
#include <cassert>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
#include <sstream>

#ifdef ACPP_HIPRTC_LINK
#define __HIP_PLATFORM_AMD__
#include <hip/hiprtc.h>
#endif


namespace hipsycl {
namespace compiler {

namespace {

const char* TargetTriple = "amdgcn-amd-amdhsa";

std::string getRocmClang(const std::string& RocmPath) {
  std::string ClangPath;

  std::string GuessedHipccPath =
      common::filesystem::join_path(RocmPath, std::vector<std::string>{"bin", "hipcc"});
  if (llvm::sys::fs::exists(GuessedHipccPath))
    ClangPath = GuessedHipccPath;
  else {
#if defined(ACPP_HIPCC_PATH)
    ClangPath = ACPP_HIPCC_PATH;
#else
    ClangPath = ACPP_CLANG_PATH;
#endif
  }

  return ClangPath;
}

#if LLVM_VERSION_MAJOR < 16
template<class T>
using optional_t = llvm::Optional<T>;
#else
template<class T>
using optional_t = std::optional<T>;
#endif

bool getCommandOutput(const std::string &Program, const llvm::SmallVector<std::string> &Invocation,
                      std::string &Out) {

  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-query-%%%%%%.txt");

  std::string OutputFilename = OutputFile->TmpName;

  auto E = OutputFile.takeError();
  if(E){
    return false;
  }

  AtScopeExit DestroyOutputFile([&]() { auto Err = OutputFile->discard(); });

  llvm::SmallVector<llvm::StringRef> InvocationRef;
  for(const auto& S: Invocation)
    InvocationRef.push_back(S);

  llvm::SmallVector<optional_t<llvm::StringRef>> Redirections;
  std::string RedirectedOutputFile = OutputFile->TmpName;
  Redirections.push_back(optional_t<llvm::StringRef>{});
  Redirections.push_back(llvm::StringRef{RedirectedOutputFile});
  Redirections.push_back(llvm::StringRef{RedirectedOutputFile});

  int R = llvm::sys::ExecuteAndWait(Program, InvocationRef, {}, Redirections); 
  if(R != 0)
    return false;

  auto ReadResult =
    llvm::MemoryBuffer::getFile(OutputFile->TmpName, true);
  
  Out = ReadResult.get()->getBuffer();
  return true;
}

}




class RocmDeviceLibs {
public:

  static bool determineRequiredDeviceLibs(const std::string& RocmPath,
                                          const std::string& DeviceLibsPath,
                                          const std::string& TargetDevice,
                                          std::vector<std::string>& BitcodeFiles,
                                          bool IsFastMath = false) {
    

    llvm::SmallVector<std::string> Invocation;
    auto OffloadArchFlag = "--cuda-gpu-arch="+TargetDevice;
    auto RocmPathFlag = "--rocm-path="+std::string{RocmPath};
    auto RocmDeviceLibsFlag = "--rocm-device-lib-path="+DeviceLibsPath;

    std::string ClangPath = getRocmClang(RocmPath);
    
    HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Invoking " << ClangPath
                       << " to determine ROCm device library list\n";

    Invocation = {ClangPath, "-x", "ir",
      "--cuda-device-only", "-O3",
      RocmPathFlag,
      RocmDeviceLibsFlag,
      OffloadArchFlag,
      "-nogpuinc",
      "/dev/null",
      "--hip-link",
      "-###"
    };
    if(IsFastMath)
      Invocation.push_back("-ffast-math");
    
    std::string Output;
    if(!getCommandOutput(ClangPath, Invocation, Output))
      return false;

    std::stringstream sstr{Output};
    std::string CurrentComponent;

    bool ConsumeNext = false;
    while(sstr) {
      sstr >> CurrentComponent;
      if(ConsumeNext) {
        ConsumeNext = false;
        if(CurrentComponent.find('\"') == 0)
          CurrentComponent = CurrentComponent.substr(1);
        if(CurrentComponent.find('\"') != std::string::npos)
          CurrentComponent = CurrentComponent.substr(0, CurrentComponent.size() - 1);
        BitcodeFiles.push_back(CurrentComponent);
      } else if(CurrentComponent == "\"-mlink-builtin-bitcode\"")
        ConsumeNext = true;
    }

    return true;
  }
};

LLVMToAmdgpuTranslator::LLVMToAmdgpuTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{sycl::jit::backend::amdgpu, KN}, KernelNames{KN} {
  RocmDeviceLibsPath = common::filesystem::join_path(RocmPath,
                                                     std::vector<std::string>{"amdgcn", "bitcode"});
}


bool LLVMToAmdgpuTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  
  M.setTargetTriple(TargetTriple);
#if LLVM_VERSION_MAJOR >= 17
  M.setDataLayout(
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-"
      "i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-"
      "n32:64-S32-A5-G1-ni:7:8");
#else
  M.setDataLayout(
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-"
      "v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");
#endif
  
  AddressSpaceMap ASMap = getAddressSpaceMap();

  KernelFunctionParameterRewriter ParamRewriter{
      // amdgpu backend wants ByRef attribute for all aggregates passed in by-value
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByRef,
      // Those pointers to by-value data should be in constant AS
      ASMap[AddressSpace::Constant],
      // Actual pointers should be in global memory
      ASMap[AddressSpace::Global]};
  
  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);
  
  for(auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Setting up kernel " << KernelName << "\n";
    if(auto* F = M.getFunction(KernelName)) {
      F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

      if(KnownGroupSizeX != 0 && KnownGroupSizeY != 0 && KnownGroupSizeZ != 0) {
        int FlatGroupSize = KnownGroupSizeX * KnownGroupSizeY * KnownGroupSizeZ;
        
        if(!F->hasFnAttribute("amdgpu-flat-work-group-size"))
          F->addFnAttr("amdgpu-flat-work-group-size",
                     std::to_string(FlatGroupSize) + "," + std::to_string(FlatGroupSize));
      }
    }
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-amdgpu-amdhsa-full.bc"});
  
  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;
  
  AddressSpaceInferencePass ASIPass {ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);
  
  // amdgpu does not like some function calls, so try to inline
  // everything. Note: This should be done after ASI pass has fixed
  // alloca address spaces, in case alloca values are passed as arguments!
  for(auto& F: M) {
    if(F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
      if(!F.empty()) {
        F.addFnAttr(llvm::Attribute::AlwaysInline);
      }
    }
  }
  llvm::AlwaysInlinerPass AIP;
  AIP.run(M, *PH.ModuleAnalysisManager);

  return true;
}


bool LLVMToAmdgpuTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &Out) {
#ifdef ACPP_HIPRTC_LINK
  HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Invoking hipRTC...\n";

  std::string ModuleString;
  llvm::raw_string_ostream StrOstream{ModuleString};
  llvm::WriteBitcodeToFile(FlavoredModule, StrOstream);

  return hiprtcJitLink(ModuleString, Out);
#else
  return clangJitLink(FlavoredModule, Out);
#endif
}

bool LLVMToAmdgpuTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "amdgpu-target-device") {
    TargetDevice = Value;
    return true;
  } else if (Option == "rocm-device-libs-path") {
    RocmDeviceLibsPath = Value;
    return true; 
  } else if (Option == "rocm-path") {
    RocmPath = Value;
    return true;
  }

  return false;
}

bool LLVMToAmdgpuTranslator::applyBuildFlag(const std::string &Flag) {

  return false;
}

bool LLVMToAmdgpuTranslator::hiprtcJitLink(const std::string &Bitcode, std::string &Output) {
#ifdef ACPP_HIPRTC_LINK
  // Currently hipRTC link does not take into account options anyway.
  // It just compiles for the currently active HIP device.
  std::vector<hiprtcJIT_option> options {};
  std::vector<void*> option_vals {};
    
  hiprtcLinkState LS;
  auto err = hiprtcLinkCreate(options.size(), options.data(),
                              option_vals.data(), &LS);
  if(err != HIPRTC_SUCCESS) {
    this->registerError("LLVMToAmdgpu: Could not create hipRTC link state");
    return false;
  }


  void* Data = static_cast<void*>(const_cast<char*>(Bitcode.data()));
  err = hiprtcLinkAddData(LS, HIPRTC_JIT_INPUT_LLVM_BITCODE, Data, Bitcode.size(),
                          "hipSYCL SSCP Bitcode", 0, 0, 0);

  auto addBitcodeFile = [&](const std::string &BCFileName) -> bool {
    std::string Path = common::filesystem::join_path(RocmDeviceLibsPath, BCFileName);
    auto ReadResult = llvm::MemoryBuffer::getFile(Path, false);
    if(auto Err = ReadResult.getError()) {
      this->registerError("LLVMToAmdgpu: Could not open file: " + Path);
      return false;
    }

    llvm::StringRef BC = ReadResult->get()->getBuffer();
    hiprtcLinkAddData(LS, HIPRTC_JIT_INPUT_LLVM_BITCODE, const_cast<char *>(BC.data()), BC.size(),
                      BCFileName.c_str(), 0, 0, 0);

    return true;
  };

  std::vector<std::string> DeviceLibs;
  RocmDeviceLibs::determineRequiredDeviceLibs(RocmPath, RocmDeviceLibsPath, TargetDevice,
                                              DeviceLibs, IsFastMath);
  for(const auto& Lib : DeviceLibs) {
    HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Linking with bitcode file: " << Lib << "\n";
    addBitcodeFile(Lib);
  }


  if(err != HIPRTC_SUCCESS) {
    this->registerError("LLVMToAmdgpu: Could not add hipRTC data for bitcode linking");
    return false;
  }

  void* Binary = nullptr;
  std::size_t Size = 0;
  err = hiprtcLinkComplete(LS, &Binary, &Size);
  if(err != HIPRTC_SUCCESS) {
    this->registerError("LLVMToAmdgpu: hiprtcLinkComplete() failed. Setting the environment "
                        "variables AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout "
                        "AMD_COMGR_EMIT_VERBOSE_LOGS=1 might reveal more information.");
    return false;
  }
    
  Output.resize(Size);
  std::copy(static_cast<char *>(Binary), static_cast<char *>(Binary) + Size, Output.begin());
    
  err = hiprtcLinkDestroy(LS);
  if(err != HIPRTC_SUCCESS) {
    this->registerError("LLVMToAmdgpu: hiprtcLinkDestroy() failed");
    return false;
  }

  return true;
#else
  return false;
#endif
}

bool LLVMToAmdgpuTranslator::clangJitLink(llvm::Module& FlavoredModule, std::string& Out) {
  
  auto addBitcodeFile = [&](const std::string &BCFileName) -> bool {
    std::string Path = common::filesystem::join_path(RocmDeviceLibsPath, BCFileName);
    auto ReadResult = llvm::MemoryBuffer::getFile(Path, false);
    if(auto Err = ReadResult.getError()) {
      this->registerError("LLVMToAmdgpu: Could not open file: " + Path);
      return false;
    }

    llvm::StringRef BC = ReadResult->get()->getBuffer();
    this->linkBitcodeFile(FlavoredModule, Path, "", "", false);

    return true;
  };

  std::vector<std::string> DeviceLibs;
  RocmDeviceLibs::determineRequiredDeviceLibs(RocmPath, RocmDeviceLibsPath, TargetDevice,
                                              DeviceLibs);
  for(const auto& BC : DeviceLibs)
    addBitcodeFile(BC);

  auto InputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-amdgpu-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-amdgpu-%%%%%%.hipfb");
  auto DummyFile = llvm::sys::fs::TempFile::create("hipsycl-sscp-amdgpu-dummy-%%%%%%.cpp");

  std::string OutputFilename = OutputFile->TmpName;

  auto checkFileError = [&](auto& F) {
    auto E = F.takeError();
    if(E){
      this->registerError("LLVMToAmdgpu: Could not create temp file: "+InputFile->TmpName);
      return false;
    }
    return true;
  };

  if(!checkFileError(InputFile)) return false;
  if(!checkFileError(DummyFile)) return false;

  AtScopeExit DestroyInputFile([&]() { auto Err = InputFile->discard(); });
  AtScopeExit DestroyOutputFile([&]() { auto Err = OutputFile->discard(); });
  AtScopeExit DestroyDummyFile([&]() { auto Err = DummyFile->discard(); });
  
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};
  llvm::raw_fd_ostream DummyStream{DummyFile->FD, false};

  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();
   
  std::string DummyText = "int main() {}\n";
  DummyStream.write(DummyText.c_str(), DummyText.size());
  DummyStream.flush();

  auto OffloadArchFlag = "--cuda-gpu-arch="+TargetDevice;
  std::string ClangPath = ACPP_CLANG_PATH;

  llvm::SmallVector<std::string> Invocation = {
      ClangPath, "-x", "hip", "-O3", "-nogpuinc", OffloadArchFlag, "--cuda-device-only",
        "-Xclang", "-mlink-bitcode-file", "-Xclang", InputFile->TmpName,
        "-o",  OutputFilename, DummyFile->TmpName
  };

  llvm::SmallVector<llvm::StringRef> InvocationRef;
  for(auto &S : Invocation)
    InvocationRef.push_back(S);

  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToAmdgpu: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      InvocationRef[0], InvocationRef);

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

  Out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToAmdgpuTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

AddressSpaceMap LLVMToAmdgpuTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 5;
  ASMap[AddressSpace::Constant] = 4;
  ASMap[AddressSpace::AllocaDefault] = 5;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 4;

  return ASMap;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToAmdgpuTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToAmdgpuTranslator>(KernelNames);
}

}
}
