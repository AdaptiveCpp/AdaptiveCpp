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
#include "hipSYCL/compiler/llvm-to-backend/metal/LLVMToMetal.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>


#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>

#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ADCE.h"

#include <memory>
#include <cassert>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "Emitter.hpp"

namespace hipsycl {
namespace compiler {

namespace {

struct ReplaceIntrinsics : llvm::PassInfoMixin<ReplaceIntrinsics> {
  struct Mapping {
    std::string name;
    int argCount;
  };

  std::unordered_map<llvm::Intrinsic::ID, Mapping> mapping = {
    {llvm::Intrinsic::ctlz, {"clz", 1}},
    {llvm::Intrinsic::cttz, {"ctz", 1}},
    {llvm::Intrinsic::ctpop, {"popcount", 1}},
  };

  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager& FAM) {
    llvm::SmallVector<llvm::IntrinsicInst*, 16> Work;

    for (auto& BB : F) {
      for (auto& I : BB) {
        auto* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
        if (!II) {
          continue;
        }
        auto ID = II->getIntrinsicID();
        if (ID == llvm::Intrinsic::ctlz || ID == llvm::Intrinsic::cttz || ID == llvm::Intrinsic::ctpop) {
          Work.push_back(II);
        }
      }
    }

    bool Changed = false;
    for (auto* II : Work) {
      if (replaceCountIntrinsic(II)) {
        II->eraseFromParent();
        Changed = true;
      }
    }

    return Changed ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
  }

  bool replaceCountIntrinsic(llvm::IntrinsicInst* II) {
    auto ID = II->getIntrinsicID();
    auto mappingIt = mapping.find(ID);
    if (mappingIt == mapping.end()) {
      return false;
    }

    llvm::Value* X = II->getArgOperand(0);
    auto* Ty = X->getType();
    unsigned W;
    if (auto* IT = llvm::dyn_cast<llvm::IntegerType>(Ty)) {
      W = IT->getBitWidth();
    } else {
      return false;
    }
    if (W != 8 && W != 16 && W != 32 && W != 64) {
      return false;
    }

    std::string FnName = "__acpp_sscp_" + mappingIt->second.name + "_u" + std::to_string(W);
    llvm::Module& M = *II->getModule();
    auto* FT = llvm::FunctionType::get(Ty, {Ty}, false);
    llvm::FunctionCallee Callee = M.getOrInsertFunction(FnName, FT);
    if (auto* F = llvm::dyn_cast<llvm::Function>(Callee.getCallee())) {
      F->addFnAttr(llvm::Attribute::AlwaysInline);
      F->addFnAttr(llvm::Attribute::NoUnwind);
      F->addFnAttr(llvm::Attribute::ReadNone);
    }

    llvm::IRBuilder<> B(II);
    llvm::Value* Call = B.CreateCall(Callee, {X});
    II->replaceAllUsesWith(Call);

    return true;
  }
};

struct ExpandIntrinsics : llvm::PassInfoMixin<ExpandIntrinsics> {
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager& FAM) {
    llvm::SmallVector<llvm::IntrinsicInst*, 16> Work;

    // expand can change CFG, so we need to collect intrinsics first and then expand them in a separate loop
    for (auto& BB : F) {
      for (auto& I : BB) {
        auto* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
        if (!II) {
          continue;
        }
        Work.push_back(II);
      }
    }

    const llvm::TargetTransformInfo& TTI = FAM.getResult<llvm::TargetIRAnalysis>(F);
    llvm::ScalarEvolution* SE = nullptr;
    if (FAM.getCachedResult<llvm::ScalarEvolutionAnalysis>(F)) {
      SE = &FAM.getResult<llvm::ScalarEvolutionAnalysis>(F);
    }

    bool Changed = false;

    for (auto* II : Work) {
      auto ID = II->getIntrinsicID();
      if (auto* MC = llvm::dyn_cast<llvm::MemCpyInst>(II)) {
        llvm::expandMemCpyAsLoop(MC, TTI, SE);
        II->eraseFromParent();
        Changed = true;
      } else if (auto* MM = llvm::dyn_cast<llvm::MemMoveInst>(II)) {
        bool lowered = llvm::expandMemMoveAsLoop(MM, TTI);
        if (lowered) {
          II->eraseFromParent();
          Changed = true;
        }
      } else if (auto* MS = llvm::dyn_cast<llvm::MemSetInst>(II)) {
        llvm::expandMemSetAsLoop(MS);
        II->eraseFromParent();
        Changed = true;
      } else if (auto* MSP = llvm::dyn_cast<llvm::MemSetPatternInst>(II)) {
        llvm::expandMemSetPatternAsLoop(MSP);
        II->eraseFromParent();
        Changed = true;
#if LLVM_VERSION_MAJOR >= 21
      } else if (auto* AMC = llvm::dyn_cast<llvm::AnyMemCpyInst>(II)) {
#else
      } else if (auto* AMC = llvm::dyn_cast<llvm::AtomicMemCpyInst>(II)) {
#endif
        llvm::expandAtomicMemCpyAsLoop(AMC, TTI, SE);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::uadd_with_overflow || ID == llvm::Intrinsic::usub_with_overflow) {
        expandUaddUsubWithOverflow(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::usub_sat || ID == llvm::Intrinsic::uadd_sat) {
        expandUaddUsubSat(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::fshl || ID == llvm::Intrinsic::fshr) {
        expandFunnelShift(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::smax || ID == llvm::Intrinsic::smin) {
        expandSminSmax(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::umax || ID == llvm::Intrinsic::umin) {
        expandUminUmax(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::abs) {
        expandAbs(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::lifetime_start ||
         ID == llvm::Intrinsic::lifetime_end ||
         ID == llvm::Intrinsic::assume ||
         ID == llvm::Intrinsic::invariant_start ||
         ID == llvm::Intrinsic::invariant_end ||
         ID == llvm::Intrinsic::experimental_noalias_scope_decl)
      {
        II->eraseFromParent();
        Changed = true;
      }
    }

    return Changed ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
  }

  void expandAbs(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* X = II->getArgOperand(0);
    llvm::Value* IsPoison = II->getArgOperand(1);

    auto* Ty = llvm::cast<llvm::IntegerType>(X->getType());

    llvm::Value* Zero = llvm::ConstantInt::get(Ty, 0);
    llvm::Value* IsNeg = B.CreateICmpSLT(X, Zero, "isneg");
    llvm::Value* Neg = B.CreateNeg(X, "neg");
    llvm::Value* Abs = B.CreateSelect(IsNeg, Neg, X, "abs");

    if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(IsPoison)) {
      if (CI->isOne()) {
        // poison on INT_MIN
        II->replaceAllUsesWith(Abs);
        return;
      }
    }

    llvm::Value* IntMin = llvm::ConstantInt::getSigned(
        Ty, -(1LL << (Ty->getBitWidth() - 1)));
    llvm::Value* IsMin = B.CreateICmpEQ(X, IntMin, "ismin");
    llvm::Value* Res = B.CreateSelect(IsMin, IntMin, Abs, "abs_safe");

    II->replaceAllUsesWith(Res);
  }

  void expandSminSmax(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    auto* Ty = A->getType();
    llvm::Value* Cmp = B.CreateICmpSGT(A, Bv, "scmp");

    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::smax) {
      Res = B.CreateSelect(Cmp, A, Bv, "smax");
    } else {
      Res = B.CreateSelect(Cmp, Bv, A, "smin");
    }

    II->replaceAllUsesWith(Res);
  }

  void expandUminUmax(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    llvm::Value* Cmp = B.CreateICmpUGT(A, Bv, "ucmp");

    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::umax) {
      Res = B.CreateSelect(Cmp, A, Bv, "umax");
    } else {
      Res = B.CreateSelect(Cmp, Bv, A, "umin");
    }

    II->replaceAllUsesWith(Res);
  }

  void expandFunnelShift(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    auto* A = II->getArgOperand(0);
    auto* Bv = II->getArgOperand(1);
    auto* S = II->getArgOperand(2);

    auto* Ty = llvm::cast<llvm::IntegerType>(A->getType());
    unsigned W = Ty->getBitWidth();

    auto* WConst = llvm::ConstantInt::get(Ty, W);
    auto* Zero   = llvm::ConstantInt::get(Ty, 0);

    llvm::Value* Shift = S;
    if (Shift->getType() != Ty) {
      Shift = B.CreateZExtOrTrunc(Shift, Ty);
    }

    llvm::Value* Sh = B.CreateURem(Shift, WConst, "sh");

    llvm::Value* IsZero = B.CreateICmpEQ(Sh, Zero, "sh_is_zero");

    llvm::Value* WmSh = B.CreateSub(WConst, Sh, "w_minus_sh");

    llvm::Value* ResShifted = nullptr;
    if (II->getIntrinsicID() == llvm::Intrinsic::fshl) {
      llvm::Value* L = B.CreateShl(A, Sh, "l");
      llvm::Value* R = B.CreateLShr(Bv, WmSh, "r");
      ResShifted = B.CreateOr(L, R, "fshl");
    } else { // fshr
      llvm::Value* L = B.CreateLShr(A, Sh, "l");
      llvm::Value* R = B.CreateShl(Bv, WmSh, "r");
      ResShifted = B.CreateOr(L, R, "fshr");
    }

    llvm::Value* Res = B.CreateSelect(IsZero, A, ResShifted, "fsh");
    II->replaceAllUsesWith(Res);
  }

  void expandUaddUsubSat(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    auto* A = II->getArgOperand(0);
    auto* Bv = II->getArgOperand(1);
    auto* Ty = A->getType();
    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::usub_sat) {
      llvm::Value* Diff = B.CreateSub(A, Bv, "diff");
      llvm::Value* Under = B.CreateICmpULT(A, Bv, "under");
      llvm::Value* Zero = llvm::ConstantInt::get(Ty, 0);
      Res = B.CreateSelect(Under, Zero, Diff, "usub.sat");
    } else {
      llvm::Value* Sum = B.CreateAdd(A, Bv, "sum");
      llvm::Value* Over = B.CreateICmpULT(Sum, A, "over");
      llvm::Value* Max = llvm::ConstantInt::getAllOnesValue(Ty);
      Res = B.CreateSelect(Over, Max, Sum, "uadd.sat");
    }
    II->replaceAllUsesWith(Res);
  }

  void expandUaddUsubWithOverflow(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    llvm::Value* Result;
    llvm::Value* Overflow;

    if (II->getIntrinsicID() == llvm::Intrinsic::uadd_with_overflow) {
      Result = B.CreateAdd(A, Bv, "sum");
      Overflow = B.CreateICmpULT(Result, A, "overflow");
    } else { // usub
      Result = B.CreateSub(A, Bv, "diff");
      Overflow = B.CreateICmpULT(A, Bv, "overflow");
    }

    llvm::Type* RetTy = II->getType();
    llvm::Value* Agg = llvm::UndefValue::get(RetTy);
    Agg = B.CreateInsertValue(Agg, Result, 0);
    Agg = B.CreateInsertValue(Agg, Overflow, 1);

    II->replaceAllUsesWith(Agg);
  }
};

} // namespace


LLVMToMetalTranslator::LLVMToMetalTranslator(const std::vector<std::string>& KernelNames)
  : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::metal), KernelNames, KernelNames}
  , KernelNames(KernelNames)
  , ActualKernelNames(KernelNames.begin(), KernelNames.end())
{ }

LLVMToMetalTranslator::~LLVMToMetalTranslator() = default;

AddressSpaceMap LLVMToMetalTranslator::getAddressSpaceMap() const
{
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

bool LLVMToMetalTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return ActualKernelNames.count(F.getName().str()) > 0;
}

bool LLVMToMetalTranslator::prepareBackendFlavor(llvm::Module& M) {
  return true;
}

bool LLVMToMetalTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  AddressSpaceMap ASMap = getAddressSpaceMap();

  AddressSpaceInferencePass ASIPass{ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  withPassBuilder([&](auto& PB, auto& LAM, auto& FAM, auto& CGAM, auto& MAM) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(ReplaceIntrinsics());
    llvm::ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.run(M, MAM);
    return 0;
  });

  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-metal-full.bc");

  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  llvm::StripDebugInfo(M);

  return true;
}

bool LLVMToMetalTranslator::translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) {
  withPassBuilder([&](auto& PB, auto& LAM, auto& FAM, auto& CGAM, auto& MAM) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::PromotePass());
    FPM.addPass(ExpandIntrinsics());
    FPM.addPass(llvm::LowerSwitchPass());
    FPM.addPass(llvm::LoopSimplifyPass());
    FPM.addPass(llvm::LCSSAPass());
    FPM.addPass(llvm::DCEPass());
    FPM.addPass(llvm::ADCEPass());
    FPM.addPass(llvm::StructurizeCFGPass());
    FPM.addPass(llvm::SimplifyCFGPass());
    llvm::ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.run(FlavoredModule, MAM);
    return 0;
  });

  std::unordered_set<std::string> kernelNames(KernelNames.begin(), KernelNames.end());

#ifdef ACPP_PRINT_IR_BEFORE_EMIT
  FlavoredModule.print(llvm::errs(), nullptr);
#endif

  MetalEmitterOptions emitterOpts;
  if (MaxArgsForFlatMode.has_value()) {
    emitterOpts.maxArgsForFlatMode = MaxArgsForFlatMode.value();
  }
  MetalEmitter emitter(FlavoredModule, kernelNames, emitterOpts);
  bool success = emitter.emit(out);
  if (!success) {
    registerError("LLVMToMetal: MetalEmitter failed: " +
                  emitter.errorMessage().value_or("unknown error"));
    return false;
  }

#ifdef ACPP_PRINT_METAL_CODE
  std::cerr << "Generated Metal code:\n" << out << std::endl;
#endif
  return true;
}

bool LLVMToMetalTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if (Option == "metal-max-args-for-flat-mode") {
    MaxArgsForFlatMode = std::stoi(Value);
    return true;
  }
  return false;
}

void LLVMToMetalTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  ActualKernelNames.erase(From->getName().str());
  ActualKernelNames.insert(To->getName().str());
}


std::unique_ptr<LLVMToBackendTranslator>
createLLVMToMetalTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToMetalTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl