/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
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
#include <llvm/IR/Metadata.h>
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
    ClangPath = getClangPath();
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

  bool Create = true;
  auto consumeError = [&](std::error_code EC) {
    if(EC) {
      if(Create)
        HIPSYCL_DEBUG_WARNING << "LLVMToAmdgpu: Could not create temp file: " << EC.message() << "\n";
      else
        HIPSYCL_DEBUG_WARNING << "LLVMToAmdgpu: Could not delete temp file: " << EC.message() << "\n";
      return false;
    }
    return true;
  };

  llvm::SmallVector<char> OutputFile;
  if(!consumeError(llvm::sys::fs::createTemporaryFile("acpp-sscp-query", "txt", OutputFile, llvm::sys::fs::OF_None))) return false;
  std::string OutputFilename = OutputFile.data();
  
  Create = false;
  AtScopeExit DestroyOutputFile([&]() { consumeError(llvm::sys::fs::remove(OutputFilename)); });

  llvm::SmallVector<llvm::StringRef> InvocationRef;
  for(const auto& S: Invocation)
    InvocationRef.push_back(S);

  llvm::SmallVector<optional_t<llvm::StringRef>> Redirections;
  std::string RedirectedOutputFile = OutputFilename;
  Redirections.push_back(optional_t<llvm::StringRef>{});
  Redirections.push_back(llvm::StringRef{RedirectedOutputFile});
  Redirections.push_back(llvm::StringRef{RedirectedOutputFile});

  int R = llvm::sys::ExecuteAndWait(Program, InvocationRef, {}, Redirections); 
  if(R != 0)
    return false;

  auto ReadResult =
    llvm::MemoryBuffer::getFile(OutputFilename, true);
  
  Out = ReadResult.get()->getBuffer();
  return true;
}

}




class RocmDeviceLibs {
private:
  static std::string extractISAAsString(const std::string &TargetDevice) {
    std::string Result = TargetDevice;
    // First remove the subtarget in strings like gfxABC:xnack-:sramecc-
    // So, find first : and throw away the stuff afterwards to obtain gfxABC
    auto ColonPos = Result.find(":");
    if(ColonPos != std::string::npos)
      Result = Result.substr(0, ColonPos);

    // Remove gfx prefix
    if(Result.find("gfx") != 0)
      return "";
    return Result.substr(3);
  }

  static int extractABIVersion(const std::string& ABIVersionLib) {
    std::string Result = ABIVersionLib;
    if(Result.find("oclc_abi_version_") != 0)
      return -1;
    else {
      Result = Result.substr(std::string{"oclc_abi_version_"}.length());
      auto DotPos = Result.find(".");
      if(DotPos == std::string::npos)
        return -1;
      
      Result = Result.substr(0, DotPos);
      return std::stoi(Result);
    }
  }

  static std::string getDefaultABIVersionLib(const std::string& DeviceLibDir) {
    std::error_code EC;
    std::vector<std::string> Files = common::filesystem::list_regular_files(DeviceLibDir, EC);
    for(auto& S : Files) {
      S = common::filesystem::filename(S);
    }
    if(EC)
      return "";
    
    std::vector<int> AvailableCodeObjectModels;
    for(const auto& F : Files) {
      int ABI = extractABIVersion(F);
      if(ABI > 0) {
        AvailableCodeObjectModels.push_back(ABI);
      }
    }
    if(AvailableCodeObjectModels.empty())
      return "";

    return "oclc_abi_version_" +
           std::to_string(*std::max_element(AvailableCodeObjectModels.begin(),
                                            AvailableCodeObjectModels.end())) +
           ".bc";
  }

public:

  static std::string getDeviceLibDirectory() {
    static std::string Path;
    if(!Path.empty())
      return Path;
    
    std::string RedistPackagePath = getRedistPackageBitcodePath("amdgcn");
    if (common::filesystem::exists(common::filesystem::join_path(RedistPackagePath, "ockl.bc")))
      Path = RedistPackagePath;
    else Path = ACPP_ROCM_DEVICE_LIBS_PATH;

    return Path;
  }

  static bool determineRequiredDeviceLibs(const std::string& TargetDevice,
                                          std::vector<std::string>& BitcodeFiles,
                                          bool IsFastMath = false,
                                          int WavefrontSize = 64,
                                          int ForceCodeObjectModel = -1) {

    if(WavefrontSize != 64 && WavefrontSize != 32)
      return false;

    std::string DeviceLibPath = getDeviceLibDirectory();
    std::string ISA = extractISAAsString(TargetDevice);
    if(ISA.empty())
      return false;

    std::vector<std::string> NeededBitcodeLibs = {
      "hip.bc",
      "ockl.bc",
      "ocml.bc",
      "oclc_isa_version_"+ISA+".bc"
    };

    if(WavefrontSize == 64) {
      NeededBitcodeLibs.push_back("oclc_wavefrontsize64_on.bc");
    } else {
      NeededBitcodeLibs.push_back("oclc_wavefrontsize64_off.bc");
    }

    // abi version
    if(ForceCodeObjectModel != -1) {
      NeededBitcodeLibs.push_back("oclc_abi_version_" + std::to_string(ForceCodeObjectModel) +
                                  ".bc");
    } else {
      std::string DefaultABILib = getDefaultABIVersionLib(DeviceLibPath);
      NeededBitcodeLibs.push_back(DefaultABILib);
    }

    if(IsFastMath) {
      NeededBitcodeLibs.push_back("oclc_correctly_rounded_sqrt_off.bc");
      NeededBitcodeLibs.push_back("oclc_daz_opt_on.bc");
      NeededBitcodeLibs.push_back("oclc_finite_only_on.bc");
      NeededBitcodeLibs.push_back("oclc_unsafe_math_on.bc");
    } else {
      NeededBitcodeLibs.push_back("oclc_correctly_rounded_sqrt_on.bc");
      NeededBitcodeLibs.push_back("oclc_daz_opt_off.bc");
      NeededBitcodeLibs.push_back("oclc_finite_only_off.bc");
      NeededBitcodeLibs.push_back("oclc_unsafe_math_off.bc");
    }

    BitcodeFiles.clear();
    for(const auto& L : NeededBitcodeLibs) {
      std::string FullPath = common::filesystem::join_path(DeviceLibPath, L);
      BitcodeFiles.push_back(FullPath);
    }

    return true;
  }
};

LLVMToAmdgpuTranslator::LLVMToAmdgpuTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::amdgpu), KN, KN},
      KernelNames{KN} {}

bool LLVMToAmdgpuTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  
#if LLVM_VERSION_MAJOR > 20
  M.setTargetTriple(llvm::Triple(TargetTriple));
#else
  M.setTargetTriple(TargetTriple);
#endif

#if LLVM_VERSION_MAJOR >= 18
  M.setDataLayout("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:"
                  "32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                  "256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9");
#elif LLVM_VERSION_MAJOR >= 17
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
      applyKernelProperties(F);
    }
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-amdgpu-amdhsa-full.bc");
  
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

  if(llvm::Metadata* MD  = M.getModuleFlag("amdgpu_code_object_version")) {
    if(auto* V = llvm::cast<llvm::ValueAsMetadata>(MD)) {
      if (llvm::ConstantInt* CI = llvm::dyn_cast<llvm::ConstantInt>(V->getValue())) {
        if (CI->getBitWidth() <= 32) {
          CodeObjectModelVersion = CI->getSExtValue();
        }
      }
    }
  }

  return true;
}

bool LLVMToAmdgpuTranslator::translateToBackendFormat(llvm::Module &FlavoredModule,
                                                      std::string &Out) {
  if(getWavefrontSize() != 32 && getWavefrontSize() != 64) {
    this->registerError("LLVMToAmdgpu: Invalid wavefront size was requested: " +
                        std::to_string(getWavefrontSize()));
    return false;
  }

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
    auto ReadResult = llvm::MemoryBuffer::getFile(BCFileName, false);
    if(auto Err = ReadResult.getError()) {
      this->registerError("LLVMToAmdgpu: Could not open file: " + BCFileName);
      return false;
    }

    llvm::StringRef BC = ReadResult->get()->getBuffer();
    hiprtcLinkAddData(LS, HIPRTC_JIT_INPUT_LLVM_BITCODE, const_cast<char *>(BC.data()), BC.size(),
                      BCFileName.c_str(), 0, 0, 0);

    return true;
  };

  std::vector<std::string> DeviceLibs;
  RocmDeviceLibs::determineRequiredDeviceLibs(TargetDevice, DeviceLibs, IsFastMath, getWavefrontSize(),
                                              CodeObjectModelVersion);
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
    auto ReadResult = llvm::MemoryBuffer::getFile(BCFileName, false);
    if(auto Err = ReadResult.getError()) {
      this->registerError("LLVMToAmdgpu: Could not open file: " + BCFileName);
      return false;
    }

    llvm::StringRef BC = ReadResult->get()->getBuffer();
    this->linkBitcodeFile(FlavoredModule, BCFileName, "", "", false);

    return true;
  };

  std::vector<std::string> DeviceLibs;
  RocmDeviceLibs::determineRequiredDeviceLibs(TargetDevice, DeviceLibs, IsFastMath,
                                              getWavefrontSize(), CodeObjectModelVersion);
  for(const auto& BC : DeviceLibs)
    addBitcodeFile(BC);

  bool Create = true;
  auto consumeError = [&](std::error_code EC) {
    if(EC) {
      if(Create)
        this->registerError("LLVMToAmdgpu: Could not create temp file: " + EC.message());
      else
        this->registerError("LLVMToAmdgpu: Could not remove temp file: " + EC.message());
      return false;
    }
    return true;
  };

  int InputFD = 0, DummyFD = 0;
  llvm::SmallVector<char> InputFileNameBuf, DummyFileNameBuf;
  // don't use fs::TempFile, as we can't unlock the file for the clang invocation later... (Windows)
  if(!consumeError(llvm::sys::fs::createTemporaryFile("acpp-sscp-amdgpu", "bc", InputFD, InputFileNameBuf, llvm::sys::fs::OF_None))) return false;
  std::string InputFileName = InputFileNameBuf.data();

  if(!consumeError(llvm::sys::fs::createTemporaryFile("acpp-sscp-amdgpu-dummy", "cpp", DummyFD, DummyFileNameBuf, llvm::sys::fs::OF_None))) return false;
  std::string DummyFileName = DummyFileNameBuf.data();

  llvm::SmallVector<char> OutputFile;
  if(!consumeError(llvm::sys::fs::createTemporaryFile("acpp-sscp-host", "hipfb", OutputFile, llvm::sys::fs::OF_None))) return false;
  std::string OutputFileName = OutputFile.data();

  Create = false;
  AtScopeExit DestroyInputFile([&]() { consumeError(llvm::sys::fs::remove(InputFileName)); });
  AtScopeExit DestroyOutputFile([&]() { consumeError(llvm::sys::fs::remove(OutputFileName)); });
  AtScopeExit DestroyDummyFile([&]() { consumeError(llvm::sys::fs::remove(DummyFileName)); });

  {
    llvm::raw_fd_ostream InputStream{InputFD, true};
    llvm::raw_fd_ostream DummyStream{DummyFD, true};

    llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
    InputStream.flush();

    std::string DummyText = "int main() {}\n";
    DummyStream.write(DummyText.c_str(), DummyText.size());
    DummyStream.flush();
  }

  auto OffloadArchFlag = "--cuda-gpu-arch="+TargetDevice;

  llvm::SmallVector<std::string> Invocation = {
      getClangPath(), "-x", "hip", "-O3", "-nogpuinc", OffloadArchFlag, "--cuda-device-only",
        "-Xclang", "-mlink-bitcode-file", "-Xclang", InputFileName,
        "-o",  OutputFileName, DummyFileName
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
      llvm::MemoryBuffer::getFile(OutputFileName, -1);

  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToAmdgpu: Could not read result file" + Err.message());
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

void LLVMToAmdgpuTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  removeKernelProperties(From);
  applyKernelProperties(To);
}

void LLVMToAmdgpuTranslator::applyKernelProperties(llvm::Function* F) {
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  if (KnownGroupSizeX != 0 && KnownGroupSizeY != 0 && KnownGroupSizeZ != 0) {
    int FlatGroupSize = KnownGroupSizeX * KnownGroupSizeY * KnownGroupSizeZ;

    if (!F->hasFnAttribute("amdgpu-flat-work-group-size"))
      F->addFnAttr("amdgpu-flat-work-group-size",
                   std::to_string(FlatGroupSize) + "," + std::to_string(FlatGroupSize));
  }
}

void LLVMToAmdgpuTranslator::removeKernelProperties(llvm::Function* F) {
  if(F->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    F->setCallingConv(llvm::CallingConv::C);
    for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i)
      if(F->getArg(i)->hasAttribute(llvm::Attribute::ByRef))
        F->getArg(i)->removeAttr(llvm::Attribute::ByRef);
  }
  if(F->hasFnAttribute("amdgpu-flat-work-group-size"))
    F->removeFnAttr("amdgpu-flat-work-group-size");
}

int LLVMToAmdgpuTranslator::getWavefrontSize() const {
  if(DesiredSubgroupSize > 0) {
    return DesiredSubgroupSize;
  }
  return 64;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToAmdgpuTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToAmdgpuTranslator>(KernelNames);
}

}
}
