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
#include "hipSYCL/compiler/llvm-to-backend/GlobalSizesFitInI32OptPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/KnownGroupSizeOptPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
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
                   const std::string &ForcedDataLayout = "",
                   llvm::Linker::Flags Flags = llvm::Linker::Flags::LinkOnlyNeeded) {
  if(!ForcedTriple.empty())
    OtherM->setTargetTriple(ForcedTriple);
  if(!ForcedDataLayout.empty())
    OtherM->setDataLayout(ForcedDataLayout);

  // Returns true on error
  if (llvm::Linker::linkModules(M, std::move(OtherM), Flags)) {
    return false;
  }
  return true;
}

void setFastMathFunctionAttribs(llvm::Module& M) {
  auto forceAttr = [&](llvm::Function& F, llvm::StringRef Key, llvm::StringRef Value) {
    if(F.hasFnAttribute(Key)) {
      if(!F.getFnAttribute(Key).getValueAsString().equals(Value))
        F.removeFnAttr(Key);
    }
    F.addFnAttr(Key, Value);
  };

  for(auto& F : M) {
    if(!F.isIntrinsic()) {
      forceAttr(F, "approx-func-fp-math","true");
      forceAttr(F, "denormal-fp-math","preserve-sign,preserve-sign");
      forceAttr(F, "no-infs-fp-math","true");
      forceAttr(F, "no-nans-fp-math","true");
      forceAttr(F, "no-signed-zeros-fp-math","true");
      forceAttr(F, "no-trapping-math","true");
      forceAttr(F, "unsafe-fp-math","true");
    }
  }
}


}

LLVMToBackendTranslator::LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
                                                 const std::vector<std::string> &OutliningEPs,
                                                 const std::vector<std::string> &KernelNames)
    : S2IRConstantBackendId(S2IRConstantCurrentBackendId),
      OutliningEntrypoints{OutliningEPs}, Kernels{KernelNames} {}

bool LLVMToBackendTranslator::setBuildFlag(const std::string &Flag) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Using build flag: " << Flag << "\n";

  if(Flag == "global-sizes-fit-in-int") {
    GlobalSizesFitInInt = true;
    return true;
  } else if(Flag == "fast-math") {
    IsFastMath = true;
    return true;
  }

  return applyBuildFlag(Flag);
}

bool LLVMToBackendTranslator::setBuildOption(const std::string &Option, const std::string &Value) {
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Using build option: " << Option << "=" << Value << "\n";

  if(Option == "known-group-size-x") {
    KnownGroupSizeX = std::stoi(Value);
    return true;
  } else if (Option == "known-group-size-y") {
    KnownGroupSizeY = std::stoi(Value);
    return true;
  } else if (Option == "known-group-size-z") {
    KnownGroupSizeZ = std::stoi(Value);
    return true;
  } else if (Option == "known-local-mem-size") {
    KnownLocalMemSize = std::stoi(Value);
  }

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

  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Applying specializations and S2 IR constants...\n";
  for(auto& A : SpecializationApplicators) {
    HIPSYCL_DEBUG_INFO << "LLVMToBackend: Processing specialization " << A.first << "\n";
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

    // These optimizations should be run before __hipsycl_sscp_* builtins
    // are resolved, so before backend bitcode libraries are linked. We thus
    // run them prior to flavoring.
    KnownGroupSizeOptPass GroupSizeOptPass{KnownGroupSizeX, KnownGroupSizeY, KnownGroupSizeZ};
    GlobalSizesFitInI32OptPass SizesAsIntOptPass{GlobalSizesFitInInt, KnownGroupSizeX,
                                                 KnownGroupSizeY, KnownGroupSizeZ};
    GroupSizeOptPass.run(M, MAM);
    SizesAsIntOptPass.run(M, MAM);

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

      if(IsFastMath)
        setFastMathFunctionAttribs(M);
      OptimizationSuccessful = optimizeFlavoredIR(M, PH);

      if(!OptimizationSuccessful) {
        this->registerError("LLVMToBackend: Optimization failed");
      }

      S2IRConstant::forEachS2IRConstant(M, [&](S2IRConstant C) {
        if (C.isValid()) {
          if (!C.isInitialized()) {
            ContainsUnsetIRConstants = true;
            this->registerError("LLVMToBackend: AdaptiveCpp S2IR constant was not set: " +
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

  // silence optimization remarks,..
  M.getContext().setDiagnosticHandlerCallBack(
      [](const llvm::DiagnosticInfo &DI, void *Context) {
        llvm::DiagnosticPrinterRawOStream DP(llvm::errs());
        if (DI.getSeverity() == llvm::DS_Error) {
          llvm::errs() << "LLVMToBackend: Error: ";
          DI.print(DP);
          llvm::errs() << "\n";
        }
      });

  llvm::ModulePassManager MPM =
      PH.PassBuilder->buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  MPM.run(M, *PH.ModuleAnalysisManager);

  return true;
}

bool LLVMToBackendTranslator::linkBitcodeString(llvm::Module &M, const std::string &Bitcode,
                                                const std::string &ForcedTriple,
                                                const std::string &ForcedDataLayout,
                                                bool LinkOnlyNeeded) {
  std::unique_ptr<llvm::Module> OtherModule;
  auto err = loadModuleFromString(Bitcode, M.getContext(), OtherModule);

  if (err) {
    this->registerError("LLVMToBackend: Could not load LLVM module");
    llvm::handleAllErrors(std::move(err), [&](llvm::ErrorInfoBase &EIB) {
      this->registerError(EIB.message());
    });
    return false;
  }

  llvm::Linker::Flags F = llvm::Linker::None;
  if(LinkOnlyNeeded)
    F = llvm::Linker::LinkOnlyNeeded;

  if(!linkBitcode(M, std::move(OtherModule), ForcedTriple, ForcedDataLayout, F)) {
    this->registerError("LLVMToBackend: Linking module failed");
    return false;
  }

  return true;
}

bool LLVMToBackendTranslator::linkBitcodeFile(llvm::Module &M, const std::string &BitcodeFile,
                                              const std::string &ForcedTriple,
                                              const std::string &ForcedDataLayout,
                                              bool LinkOnlyNeeded) {
  auto F = llvm::MemoryBuffer::getFile(BitcodeFile);
  if(auto Err = F.getError()) {
    this->registerError("LLVMToBackend: Could not open file " + BitcodeFile);
    return false;
  }
  HIPSYCL_DEBUG_INFO << "LLVMToBackend: Linking with bitcode file: " << BitcodeFile << "\n";
  return linkBitcodeString(M, std::string{F.get()->getBuffer()}, ForcedTriple, ForcedDataLayout,
                           LinkOnlyNeeded);
}

void LLVMToBackendTranslator::setS2IRConstant(const std::string &name, const void *ValueBuffer) {
  SpecializationApplicators[name] = [=](llvm::Module& M){
    S2IRConstant C = S2IRConstant::getFromConstantName(M, name);
    C.set(ValueBuffer);
  };
}

void LLVMToBackendTranslator::specializeKernelArgument(const std::string &KernelName, int ParamIndex,
                                const void *ValueBuffer) {
  std::string Id = KernelName+"__specialized_kernel_argument_"+std::to_string(ParamIndex);
  SpecializationApplicators[Id] = [=](llvm::Module& M) {
    if(auto* F = M.getFunction(KernelName)) {
      if(F->getFunctionType()->getNumParams() > ParamIndex && !F->isDeclaration()) {
        
        llvm::Type* ParamType = F->getFunctionType()->getParamType(ParamIndex);
        if (ParamType->isIntegerTy() || ParamType->isPointerTy() || ParamType->isFloatTy() ||
            ParamType->isDoubleTy()) {
          std::string GetterName = "__specialization_getter_"+Id;
          llvm::Function *GetConstant = llvm::dyn_cast<llvm::Function>(
              M.getOrInsertFunction(GetterName, ParamType).getCallee());
          
          if(!GetConstant)
            return;
          GetConstant->addFnAttr(llvm::Attribute::AlwaysInline);

          llvm::Constant *ReturnedValue = nullptr;
          std::size_t ParamByteSize = M.getDataLayout().getTypeSizeInBits(ParamType) / CHAR_BIT;

          if(ParamType->isIntegerTy()) {
            uint64_t Value = 0;
            std::memcpy(&Value, ValueBuffer, ParamByteSize);
            ReturnedValue = llvm::ConstantInt::get(
              M.getContext(), llvm::APInt(ParamType->getIntegerBitWidth(), Value));
          } else if(ParamType->isFloatTy()) {
            float Value = 0.0f;
            std::memcpy(&Value, ValueBuffer, ParamByteSize);
            ReturnedValue = llvm::ConstantFP::get(M.getContext(), llvm::APFloat(Value));
          } else if(ParamType->isDoubleTy()) {
            double Value = 0.0;
            std::memcpy(&Value, ValueBuffer, ParamByteSize);
            ReturnedValue = llvm::ConstantFP::get(M.getContext(), llvm::APFloat(Value));
          } else if(ParamType->isPointerTy()) {
            uint64_t Value = 0;
            std::memcpy(&Value, ValueBuffer, ParamByteSize);
            auto* IntPtr = llvm::ConstantInt::get(
              M.getContext(), llvm::APInt(ParamByteSize * CHAR_BIT, Value));
            ReturnedValue = llvm::ConstantExpr::getIntToPtr(
                IntPtr, ParamType);
          }
          if(!ReturnedValue) {
            HIPSYCL_DEBUG_WARNING << "LLVMToBackend: Could not specialize kernel argument " << Id
                                  << " due to unsupported parameter type\n";
            return;
          }

          llvm::BasicBlock *BB =
              llvm::BasicBlock::Create(M.getContext(), "", GetConstant);

          llvm::ReturnInst::Create(M.getContext(), ReturnedValue, BB);

          llvm::Instruction* InsertionPt = &(*F->getEntryBlock().getFirstInsertionPt());
          auto* FnCall = llvm::CallInst::Create(llvm::FunctionCallee(GetConstant),
                                  llvm::ArrayRef<llvm::Value *>{}, "", InsertionPt);
          F->getArg(ParamIndex)->replaceNonMetadataUsesWith(FnCall);
        }
      }
    }
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

