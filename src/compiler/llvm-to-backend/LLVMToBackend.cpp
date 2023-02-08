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
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"

#include <cstdint>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <string>

namespace hipsycl {
namespace compiler {

namespace {

bool linkBitcode(llvm::Module &M, std::unique_ptr<llvm::Module> OtherM,
                   const std::string &ForcedTriple = "",
                   const std::string &ForcedDataLayout = "") {
  if(!ForcedTriple.empty())
    OtherM->setTargetTriple(ForcedTriple);
  if(!ForcedDataLayout.empty())
    OtherM->setDataLayout(ForcedDataLayout);

  // Returns true on error
  if (llvm::Linker::linkModules(M, std::move(OtherM),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    return false;
  }
  return true;
}

}

LLVMToBackendTranslator::LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
  const std::vector<std::string>& OutliningEPs)
: S2IRConstantBackendId(S2IRConstantCurrentBackendId), OutliningEntrypoints{OutliningEPs} {}

bool LLVMToBackendTranslator::setBuildFlag(const std::string &Flag) { 
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Using build flag: " << Flag << "\n";
  return applyBuildFlag(Flag);
}

bool LLVMToBackendTranslator::setBuildOption(const std::string &Option, const std::string &Value) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Using build option: " << Option << "=" << Value << "\n";
  return applyBuildOption(Option, Value);
}
bool LLVMToBackendTranslator::setBuildToolArguments(const std::string &ToolName,
                                    const std::vector<std::string> &Args) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Using tool arguments for tool " << ToolName << ":\n";
  for(const auto& A : Args) {
    HIPSYCL_DEBUG_INFO << "   " << A << "\n";
  }
  return applyBuildToolArguments(ToolName, Args);
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
  if (!prepareIR(*M)) {
    setFailedIR(*M);
    return false;
  }
  
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
  if (!prepareIR(*M)) {
    setFailedIR(*M);
    return false;
  }
  if (!translatePreparedIR(*M, out)) {
    setFailedIR(*M);
    return false;
  }

  return true;
}

bool LLVMToBackendTranslator::prepareIR(llvm::Module &M) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Preparing backend flavoring...\n";

  if(!this->prepareBackendFlavor(M))
    return false;
  
  // We need to resolve symbols now instead of after optimization, because we
  // may have to reuotline if the code that is linked in after symbol resolution
  // depends on IR constants.
  // This also means that we cannot error yet if we cannot resolve all symbols :(
  resolveExternalSymbols(M);

  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Applying S2 IR constants...\n";
  for(auto& A : S2IRConstantApplicators) {
    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Setting S2 IR constant " << A.first << "\n";
    A.second(M);
  }

  bool ContainsUnsetIRConstants = false;
  bool FlavoringSuccessful = false;
  bool OptimizationSuccessful = false;

  constructPassBuilderAndMAM([&](llvm::PassBuilder &PB, llvm::ModuleAnalysisManager &MAM) {
    PassHandler PH {&PB, &MAM};

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
    FlavoringSuccessful = this->toBackendFlavor(M, PH);

    // Before optimizing, make sure everything has internal linkage to
    // help inlining. All linking should have occured by now, except
    // for backend builtin libraries like libdevice etc
    for(auto & F : M) {
      // Ignore kernels and intrinsics
      if(!F.isIntrinsic() && !this->isKernelAfterFlavoring(F)) {
        // Ignore undefined functions
        if(!F.empty()) {
          F.setLinkage(llvm::GlobalValue::InternalLinkage);
          // Some backends (amdgpu) require inlining, for others it
          // just cleans up the code.
          if(!F.hasFnAttribute(llvm::Attribute::AlwaysInline))
            F.addFnAttr(llvm::Attribute::AlwaysInline);
        }
      }
    }

    if(FlavoringSuccessful) {
      // Run optimizations
      HIPSYCL_DEBUG_INFO << "LLVMToBackend: Optimizing flavored IR...\n";
      
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

bool LLVMToBackendTranslator::linkBitcodeString(llvm::Module &M, const std::string &Bitcode,
                                                const std::string &ForcedTriple,
                                                const std::string &ForcedDataLayout) {
  std::unique_ptr<llvm::Module> OtherModule;
  auto err = loadModuleFromString(Bitcode, M.getContext(), OtherModule);

  if (err) {
    this->registerError("LLVMToBackend: Could not load LLVM module");
    llvm::handleAllErrors(std::move(err), [&](llvm::ErrorInfoBase &EIB) {
      this->registerError(EIB.message());
    });
    return false;
  }

  if(!linkBitcode(M, std::move(OtherModule), ForcedTriple, ForcedDataLayout)) {
    this->registerError("LLVMToBackend: Linking module failed");
    return false;
  }

  return true;
}

bool LLVMToBackendTranslator::linkBitcodeFile(llvm::Module &M, const std::string &BitcodeFile,
                                              const std::string &ForcedTriple,
                                              const std::string &ForcedDataLayout) {
  auto F = llvm::MemoryBuffer::getFile(BitcodeFile);
  if(auto Err = F.getError()) {
    this->registerError("LLVMToBackend: Could not open file " + BitcodeFile);
    return false;
  }
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Linking with bitcode file: " << BitcodeFile << "\n";
  return linkBitcodeString(M, std::string{F.get()->getBuffer()}, ForcedTriple, ForcedDataLayout);
}

void LLVMToBackendTranslator::setS2IRConstant(const std::string &name, const void *ValueBuffer) {
  S2IRConstantApplicators[name] = [=](llvm::Module& M){
    S2IRConstant C = S2IRConstant::getFromConstantName(M, name);
    C.set(ValueBuffer);
  };
}

void LLVMToBackendTranslator::provideExternalSymbolResolver(ExternalSymbolResolver Resolver) {
  this->SymbolResolver = Resolver;
  this->HasExternalSymbolResolver = true;
}

void LLVMToBackendTranslator::resolveExternalSymbols(llvm::Module& M) {

  if(HasExternalSymbolResolver) {
    
    // TODO We can not rely on LinkedIRIds being reliable, since
    // we only link needed symbols. Therefore, just because we have linked one module once
    // we may have to do it again.
    llvm::SmallSet<std::string, 32> AllAttemptedSymbolResolutions;
    llvm::SmallSet<std::string, 16> UnresolvedSymbolsSet;

    // Find out which unresolved symbols are in this IR
    for(auto SymbolName : SymbolResolver.getImportedSymbols()) {
      HIPSYCL_DEBUG_INFO << "LLVMToBackend: Attempting to resolve primary symbol " << SymbolName
                         << "\n";
      UnresolvedSymbolsSet.insert(SymbolName);
    }

    for(;;) {
      std::vector<std::string> Symbols;
      for(auto S : UnresolvedSymbolsSet) {
        Symbols.push_back(S);
        AllAttemptedSymbolResolutions.insert(S);
      }

      std::vector<ExternalSymbolResolver::LLVMModuleId> IRs = SymbolResolver.mapSymbolsToModuleIds(Symbols);
      HIPSYCL_DEBUG_INFO << "LLVMToBackend: Attempting to link against " << IRs.size()
                        << " external bitcode modules to resolve " << UnresolvedSymbolsSet.size()
                        << " symbols\n";

      // It can happen that the IR we have just linked needs new, external
      // symbol definitions to work. So we need to try to resolve the new
      // stuff in the next iteration.
      llvm::SmallSet<std::string, 16> NewUnresolvedSymbolsSet;
      
      for(const auto& IRID : IRs) {

        SymbolListType NewUndefinedSymbolsFromIR;
        
        if (!this->linkBitcodeString(
                M, SymbolResolver.retrieveBitcode(IRID, NewUndefinedSymbolsFromIR))) {
          HIPSYCL_DEBUG_WARNING
              << "LLVMToBackend: Linking against bitcode to resolve symbols failed\n";
        }

        for(const auto& S : NewUndefinedSymbolsFromIR) {
          if(!AllAttemptedSymbolResolutions.contains(S)) {
            NewUnresolvedSymbolsSet.insert(S);
            HIPSYCL_DEBUG_INFO << "LLVMToBackend: Attemping resolve symbol " << S
                                << " as a dependency\n";
          }
        }
        
      }

      if(NewUnresolvedSymbolsSet.empty()) {
        return;
      }

      UnresolvedSymbolsSet = NewUnresolvedSymbolsSet;
    }
  }
}

void LLVMToBackendTranslator::setFailedIR(llvm::Module& M) {
  llvm::raw_string_ostream Stream{ErroringCode};
  llvm::WriteBitcodeToFile(M, Stream);
}

}
}

