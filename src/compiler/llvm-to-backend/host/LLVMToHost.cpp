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
#include "hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/dylib_loader.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/HostKernelWrapperPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/StaticLocalMemoryPass.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

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
#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
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

#ifdef __APPLE__

#include <sys/sysctl.h>

namespace {

std::string get_macos_version() {
  char    buff [64] = "";
  std::size_t buff_size = sizeof(buff);

  if (sysctlbyname("kern.osproductversion", buff, &buff_size, nullptr, 0) != 0) {
    return {};
  }
  return std::string{buff};
}

std::string get_macos_sdk_path() {
  auto xcrun = llvm::sys::findProgramByName("xcrun");
  if(!xcrun) return {};

  llvm::SmallVector<char, 64> tmpFile;
  int fd = -1;
  if(auto ec = llvm::sys::fs::createTemporaryFile("acpp-xcrun", "txt", fd, tmpFile))
    return {};
  llvm::StringRef tmpName(tmpFile.data());

  // xcrun --show-sdk-path > tmp
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects;
  redirects.push_back(std::nullopt);      // stdin
  redirects.push_back(tmpName);           // stdout -> file
  redirects.push_back(std::nullopt);      // stderr

  llvm::SmallVector<llvm::StringRef, 4> args{
    *xcrun, "--show-sdk-path"
  };

  int rc = llvm::sys::ExecuteAndWait(*xcrun, args, std::nullopt, redirects);
  if(rc != 0) {
    llvm::sys::fs::remove(tmpName);
    return {};
  }

  auto bufOrErr = llvm::MemoryBuffer::getFile(tmpName);
  llvm::sys::fs::remove(tmpName);
  if(!bufOrErr) return {};

  std::string s = bufOrErr.get()->getBuffer().str();

  // trim whitespace/newline
  while(!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  return s;
}

}

#endif

namespace hipsycl {
namespace compiler {

#if LLVM_VERSION_MAJOR >= 16
#define NULLOPT std::nullopt
#define OPTIONAL std::optional
#else
#define NULLOPT llvm::None
#define OPTIONAL llvm::Optional
#endif

LLVMToHostTranslator::LLVMToHostTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::host), KN, KN},
      KernelNames{KN} {}

bool LLVMToHostTranslator::toBackendFlavor(llvm::Module &M, PassHandler &PH) {
#ifdef _WIN32
  // Remove /DEFAULTLIB and co.
  if(auto LinkerOptionsMD = M.getNamedMetadata("llvm.linker.options")) {
    LinkerOptionsMD->eraseFromParent();
  }
#endif

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

#ifdef _WIN32
      // Windows exceptions..
      F->setPersonalityFn(nullptr);
#endif
    }
  }

  // This pass needs to be run before builtins are linked,
  // as it potentially generates additional builtin calls.
  // So we cannot run it in the pipeline at the end of this function.
  HostStaticLocalMemoryPass SLMPass{};
  SLMPass.run(M, *PH.ModuleAnalysisManager);

  std::string BuiltinBitcodeFileName = "libkernel-sscp-host-full.bc";
  if(IsFastMath)
    BuiltinBitcodeFileName = "libkernel-sscp-host-fast-full.bc";
  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), BuiltinBitcodeFileName);

  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  // Internalize all constant global variables that don't their definition
  // to be imported from external sources - this is fine because llvm-to-backend
  // lowering always happens *after* linking in all dependencies, and therefore
  // no symbols need to be exported to other TUs, ever.
  for(auto& GV : M.globals()) {
    if (GV.isConstant() && GV.hasInitializer() &&
        !llvmutils::starts_with(GV.getName(), "__acpp_cbs"))
      GV.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }

  llvm::ModulePassManager MPM;
  PH.ModuleAnalysisManager->clear(); // for some reason we need to reset the analyses... otherwise
                                     // we get a crash at IPSCCP

  PH.PassBuilder->registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager &MAM) {
    MAM.registerPass([] { return SplitterAnnotationAnalysis{}; });
  });
  PH.PassBuilder->registerModuleAnalyses(*PH.ModuleAnalysisManager);

  registerCBSPipeline(MPM, hipsycl::compiler::OptLevel::O3, true);
  HIPSYCL_DEBUG_INFO << "LLVMToHostTranslator: Done registering\n";

  llvm::FunctionPassManager FPM;
  FPM.addPass(HostKernelWrapperPass{KnownLocalMemSize, KnownGroupSizeX, KnownGroupSizeY, KnownGroupSizeZ});
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(M, *PH.ModuleAnalysisManager);
  HIPSYCL_DEBUG_INFO << "LLVMToHostTranslator: Done toBackendFlavor\n";
  return true;
}

bool LLVMToHostTranslator::translateToBackendFormat(llvm::Module &FlavoredModule,
                                                    std::string &out) {

  llvm::SmallVector<char> InputFile;
  int InputFD;
  // don't use fs::TempFile, as we can't unlock the file for the clang invocation later... (Windows)
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host", "bc", InputFD, InputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp input file" + E.message());
    return false;
  }
  llvm::StringRef InputFileName = InputFile.data();

  AtScopeExit RemoveInputFile([&](){auto Err = llvm::sys::fs::remove(InputFileName);});

  llvm::SmallVector<char> OptOutputFile;
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host-opt", "bc", OptOutputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp file" + E.message());
    return false;
  }
  llvm::StringRef OptOutputFileName = OptOutputFile.data();
  AtScopeExit RemoveOptOutputFile([&](){auto Err = llvm::sys::fs::remove(OptOutputFileName);});

  llvm::SmallVector<char> LlcOutputFile;
#ifndef _WIN32
  std::string ObjectFileEnding = "o";
#else
  std::string ObjectFileEnding = "obj";
#endif
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host-llc", ObjectFileEnding, LlcOutputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp file" + E.message());
    return false;
  }
  llvm::StringRef LlcOutputFileName = LlcOutputFile.data();
  AtScopeExit RemoveLlcOutputFile([&](){auto Err = llvm::sys::fs::remove(LlcOutputFileName);});

  llvm::SmallVector<char> OutputFile;
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host", ACPP_SHARED_LIBRARY_EXTENSION, OutputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp input file" + E.message());
    return false;
  }
  llvm::StringRef OutputFileName = OutputFile.data();
  AtScopeExit RemoveOutputFile([&](){auto Err = llvm::sys::fs::remove(OutputFileName);});

  {
    llvm::raw_fd_ostream InputStream{InputFD, true};

    llvm::WriteBitcodeToFile(FlavoredModule, InputStream);

    if(InputStream.error()) {HIPSYCL_DEBUG_ERROR << "Error while writing" << InputStream.error().message() << '\n'; }
    InputStream.flush();
    if(InputStream.error()) {HIPSYCL_DEBUG_ERROR << "Error while flushing" << InputStream.error().message() << '\n'; }
  }

  const std::string OptPath = getOptPath();
  const std::string LLCPath = getLLCPath();
  const std::string LLDPath = getLLDPath();

  const std::string LlcCpuFlag = ACPP_LLC_HOST_CPU_FLAG;
  const std::string OptCpuFlag = ACPP_OPT_HOST_CPU_FLAG;


  llvm::SmallVector<llvm::StringRef, 16> OptInvocation{OptPath,
                                                    "-O3",
                                                    "-o",
                                                    OptOutputFileName,
                                                    InputFileName,
                                                    };

  if(!OptCpuFlag.empty())
    OptInvocation.push_back(OptCpuFlag);

  llvm::SmallVector<llvm::StringRef, 16> LlcInvocation{LLCPath,
                                                    "-O3",
                                                    "-filetype=obj",
                                                    #ifndef _WIN32
                                                    "--relocation-model=pic",
                                                    #endif
                                                    "-o",
                                                    LlcOutputFileName,
                                                    OptOutputFileName,
                                                    };

  if(!LlcCpuFlag.empty())
    LlcInvocation.push_back(LlcCpuFlag);

  if(IsFastMath) {
    LlcInvocation.push_back("--enable-unsafe-fp-math");
    LlcInvocation.push_back("--enable-no-infs-fp-math");
    LlcInvocation.push_back("--enable-no-nans-fp-math");
    LlcInvocation.push_back("--enable-no-signed-zeros-fp-math");
    LlcInvocation.push_back("--enable-no-trapping-fp-math");
  }


#ifdef __APPLE__
  static std::string os_version = get_macos_version();
  static std::string sdk_path   = get_macos_sdk_path();
  if (sdk_path.empty()) {
    this->registerError("LLVMToHost: Could not determine macOS SDK path or version: "
                        "ensure that Xcode command line tools are installed (run xcode-select --install).");
    return false;
  }
  llvm::SmallVector<llvm::StringRef, 16> LldInvocation{LLDPath,
                                                    "-dynamic",
                                                    "-dylib",
                                                    "-undefined", "dynamic_lookup",
#ifdef __arm64__
                                                    "-arch","arm64",
#else
                                                    "-arch", "x86_64",
#endif                                              // TODO Figure out platform version programmatically
                                                    "-platform_version","macos", os_version, os_version,
                                                    "-mllvm", "-enable-linkonceodr-outlining",
                                                    "-syslibroot", sdk_path,
                                                    "-o",
                                                    OutputFileName,
                                                    LlcOutputFileName,
                                                    "-lSystem", // needed to prevent error 'missing LC_LOAD_DYLIB (must link with at least libSystem.dylib'
                                                    };
#elif defined(_WIN32)
  std::string LldOutputFlag = "/out:"+OutputFileName.str();
  llvm::SmallVector<llvm::StringRef, 16> LldInvocation{LLDPath,
                                                    "/dll",
                                                    "/noimplib",
                                                    "/defaultlib:libcmt",
                                                    "/defaultlib:oldnames",
                                                    LldOutputFlag,
                                                    LlcOutputFileName
                                                    };
#else
  llvm::SmallVector<llvm::StringRef, 16> LldInvocation{LLDPath,
                                                    "-shared",
                                                    "-o",
                                                    OutputFileName,
                                                    LlcOutputFileName,
                                                    };
#endif
  const llvm::StringRef AdditionalLlcFlags = ACPP_LLC_ADDITIONAL_FLAGS;
  const llvm::StringRef AdditionalOptFlags = ACPP_OPT_ADDITIONAL_FLAGS;
  AdditionalLlcFlags.split(LlcInvocation, ' ', -1, false);
  AdditionalOptFlags.split(OptInvocation, ' ', -1, false);

  auto getInvocationAsString = [](const auto& I) {
    std::string S;
    for(const auto& Arg : I) {
      S += Arg;
      S += " ";
    }
    return S;
  };


  llvm::SmallVector<OPTIONAL<llvm::StringRef>> Redirects;
  if(hipsycl::common::output_stream::get().get_debug_level() < 3) {
    // This suppresses vectorization failure warnings, which are unavoidable
    // for some code patterns. Unfortunately, --no-warn and similar seem to be
    // insufficient.
    // When an empty redirect is used, then LLVM redirects output to /dev/null
    // (or similar)
    static const char EmptyRedirect [] = "";

    for(int i = 0; i < 3; ++i)
      Redirects.push_back(llvm::StringRef{EmptyRedirect});
  }

  HIPSYCL_DEBUG_INFO << "LLVMToHost: Invoking " << getInvocationAsString(OptInvocation) << "\n";
  int R = llvm::sys::ExecuteAndWait(OptPath, OptInvocation, NULLOPT, Redirects);

  if (R != 0) {
    this->registerError("LLVMToHost: opt invocation failed with exit code " + std::to_string(R));
    return false;
  }

  HIPSYCL_DEBUG_INFO << "LLVMToHost: Invoking " << getInvocationAsString(LlcInvocation) << "\n";
  R = llvm::sys::ExecuteAndWait(LLCPath, LlcInvocation, NULLOPT, Redirects);

  if (R != 0) {
    this->registerError("LLVMToHost: llc invocation failed with exit code " + std::to_string(R));
    return false;
  }

  R = llvm::sys::ExecuteAndWait(LLDPath, LldInvocation, NULLOPT, Redirects);

  if (R != 0) {
    this->registerError("LLVMToHost: lld invocation failed with exit code " + std::to_string(R));
    return false;
  }

  auto ReadResult = llvm::MemoryBuffer::getFile(OutputFileName);

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
  // Zero initialize for CPU.. we don't have address spaces
  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 0;
  ASMap[AddressSpace::Local] = 0;
  ASMap[AddressSpace::Private] = 0;
  ASMap[AddressSpace::Constant] = 0;
  ASMap[AddressSpace::AllocaDefault] = 0;
  ASMap[AddressSpace::GlobalVariableDefault] = 0;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 0;

  return ASMap;
}

ACPP_BACKEND_API_EXPORT std::unique_ptr<LLVMToBackendTranslator>
createLLVMToHostTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToHostTranslator>(KernelNames);
}

void LLVMToHostTranslator::migrateKernelProperties(llvm::Function *From, llvm::Function *To) {
  assert(false && "migrateKernelProperties is unsupported for LLVMToHost");
}

} // namespace compiler
} // namespace hipsycl
