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
#include <llvm/Transforms/IPO/AlwaysInliner.h>

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

// these are remapped for f32 and f64
static constexpr std::array remapped_llvm_math_builtins = {
  "sin", "cos", "tan", "sqrt",
  "asin", "acos", "atan", "atan2",
  "sinh", "cosh", "tanh",
  "log", "log2", "log10",
  "exp", "exp2", "exp10",
  "ldexp",
  "fabs", "floor", "ceil",
  "copysign"
};

using builtin_mapping = std::tuple<const char*, const char*, int>; // llvm name, acpp name, arg count (for count intrinsics)

// LLVM math intrinsics where the ACPP name differs from the LLVM name
// Format: {llvm_name, acpp_name}
// llvm.<llvm_name>.f32 -> __acpp_sscp_<acpp_name>_f32
static constexpr std::array remapped_llvm_math_builtins_renamed = {
  builtin_mapping{"maxnum", "fmax", -1},
  builtin_mapping{"minnum", "fmin", -1},
  builtin_mapping{"fmuladd", "fma", -1},
  builtin_mapping{"pow", "powr", -1},
};

static constexpr std::array remapped_llvm_count_builtins = {
  builtin_mapping{"ctlz", "clz", 1},
  builtin_mapping{"cttz", "ctz", 1},
  builtin_mapping{"ctpop", "popcount", 1}
};

struct ReplaceIntrinsics : llvm::PassInfoMixin<ReplaceIntrinsics> {
  std::unordered_map<std::string, std::pair<std::string, int>> Replacement;

  ReplaceIntrinsics() {
    std::string llvm_prefix = "llvm.";
    std::string acpp_prefix = "__acpp_sscp_";
    for (const auto& Name : remapped_llvm_math_builtins) {
      Replacement[llvm_prefix + Name + ".f32"] = {acpp_prefix + Name + "_f32", -1};
      Replacement[llvm_prefix + Name + ".f64"] = {acpp_prefix + Name + "_f64", -1};
    }
    for (const auto& [Name, Mapping, ArgCount] : remapped_llvm_math_builtins_renamed) {
      Replacement[llvm_prefix + Name + ".f32"] = {acpp_prefix + Mapping + "_f32", ArgCount};
      Replacement[llvm_prefix + Name + ".f64"] = {acpp_prefix + Mapping + "_f64", ArgCount};
    }
    Replacement["llvm.powi.f32.i32"] = {acpp_prefix + "pown_f32", -1};
    Replacement["llvm.powi.f32.i64"] = {acpp_prefix + "pown_f32", -1};
    Replacement["__assert_rtn"] = {"__acpp_sscp_assert_fail", -1};
    for (const auto& [Name, Mapping, ArgCount] : remapped_llvm_count_builtins) {
      Replacement[llvm_prefix + Name + ".i8"] = {acpp_prefix + Mapping + "_u8", ArgCount};
      Replacement[llvm_prefix + Name + ".i16"] = {acpp_prefix + Mapping + "_u16", ArgCount};
      Replacement[llvm_prefix + Name + ".i32"] = {acpp_prefix + Mapping + "_u32", ArgCount};
      Replacement[llvm_prefix + Name + ".i64"] = {acpp_prefix + Mapping + "_u64", ArgCount};
    }
  }

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    for(const auto& [Name, Value] : Replacement) {
      const auto& [ReplacementName, ArgCount] = Value;
      if(llvm::Function* F = M.getFunction(Name)) {
        llvm::Function* Replacement = M.getFunction(ReplacementName);
        if(!Replacement) {
          if (ArgCount == -1) {
            Replacement = llvm::Function::Create(F->getFunctionType(), F->getLinkage(), ReplacementName, M);
          } else {
            llvm::Type* RetTy = F->getReturnType();
            llvm::SmallVector<llvm::Type*, 8> ArgTys;
            for (unsigned i = 0; i < ArgCount; ++i) {
              ArgTys.push_back(F->getArg(i)->getType());
            }
            llvm::FunctionType* FT = llvm::FunctionType::get(RetTy, ArgTys, false);
            Replacement = llvm::Function::Create(FT, F->getLinkage(), ReplacementName, M);
          }
          Replacement->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }

        HIPSYCL_DEBUG_INFO << "Metal: ReplaceIntrinsics: Remapping calls from " << Name << " to "
                           << ReplacementName << "\n";
        if (F->getFunctionType() == Replacement->getFunctionType()) {
          F->replaceAllUsesWith(Replacement);
        } else {
          // Signatures differ (e.g. llvm.ctlz has an extra i1 is_zero_undef arg)
          llvm::SmallVector<llvm::CallInst*, 16> Calls;
          for (auto* U : F->users()) {
            if (auto* CI = llvm::dyn_cast<llvm::CallInst>(U)) {
              Calls.push_back(CI);
            }
          }
          for (auto* CI : Calls) {
            llvm::SmallVector<llvm::Value*, 4> Args;
            for (unsigned i = 0; i < (unsigned)ArgCount; ++i) {
              Args.push_back(CI->getArgOperand(i));
            }
            llvm::CallInst* NewCI = llvm::CallInst::Create(Replacement->getFunctionType(), Replacement, Args, "", CI->getIterator());
            NewCI->takeName(CI);
            CI->replaceAllUsesWith(NewCI);
            CI->eraseFromParent();
          }
        }
      }
    }

    return llvm::PreservedAnalyses::none();
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
      } else if (ID == llvm::Intrinsic::scmp || ID == llvm::Intrinsic::ucmp) {
        expandCmpIntrinsic(II);
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

  void expandCmpIntrinsic(llvm::IntrinsicInst* II) {
    auto* CI = llvm::cast<llvm::CmpIntrinsic>(II);
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = CI->getLHS();
    llvm::Value* Bv = CI->getRHS();
    auto* RetTy = II->getType();

    llvm::Value* GT = B.CreateICmp(CI->getGTPredicate(), A, Bv, "gt");
    llvm::Value* LT = B.CreateICmp(CI->getLTPredicate(), A, Bv, "lt");

    auto* One    = llvm::ConstantInt::get(RetTy, 1);
    auto* Zero   = llvm::ConstantInt::get(RetTy, 0);
    auto* MinOne = llvm::ConstantInt::getSigned(RetTy, -1);

    llvm::Value* Res = B.CreateSelect(GT, One, B.CreateSelect(LT, MinOne, Zero));
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

  // First linking: provides __acpp_sscp_* definitions so that the base class inliner
  // (which runs after toBackendFlavor) can inline them. The inlined bodies then go through
  // the base class O3 optimization pipeline, which may re-introduce LLVM intrinsics such as
  // llvm.minnum / llvm.maxnum / llvm.fmuladd via InstCombine. Those are handled in
  // translateToBackendFormat with a second ReplaceIntrinsics + link pass.
  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-metal-full.bc");
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  llvm::StripDebugInfo(M);

  return true;
}

bool LLVMToMetalTranslator::translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) {
  auto ok = withPassBuilder([&](auto& PB, auto& LAM, auto& FAM, auto& CGAM, auto& MAM) {
    // Second ReplaceIntrinsics + link pass: the base class O3 pipeline (InstCombine etc.) may
    // have re-introduced LLVM intrinsics (llvm.minnum, llvm.maxnum, llvm.fmuladd) from the
    // inlined builtin bodies. We remap them to __acpp_sscp_* builtins here, then re-link the
    // Metal bitcode to supply their definitions. The subsequent inliner pass (AlwaysInlinerPass
    // inside withPassBuilder) inlines those definitions so MetalEmitter can see the
    // __acpp_sscp_metal_math_* calls it needs to emit native Metal code.
    // Any LLVM intrinsics that remain after linking have no __acpp_sscp_* counterpart and are
    // lowered to plain IR by ExpandIntrinsics below.
    ReplaceIntrinsics{}.run(FlavoredModule, MAM);

    std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-metal-full.bc");

    if (!linkBitcodeFile(FlavoredModule, BuiltinBitcodeFile))
      return false;

    llvm::AlwaysInlinerPass{}.run(FlavoredModule, MAM);

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
    return true;
  });

  if (!ok) {
    registerError("LLVMToMetal: Failed to prepare module for Metal translation");
    return false;
  }

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