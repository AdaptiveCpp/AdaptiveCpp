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

#include "hipSYCL/compiler/llvm-to-backend/metal/IntegerLegalizationPass.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>

#include <set>
#include <sstream>

namespace hipsycl {
namespace compiler {

namespace {
  static const std::set<unsigned> allowedWidths = {1, 8, 16, 32, 64, 128};

  bool isIllegalInt(const llvm::Type* Ty) {
    if (!Ty->isIntegerTy()) {
      return false;
    }
    return allowedWidths.count(Ty->getIntegerBitWidth()) == 0;
  }

  unsigned promoteWidth(unsigned N) {
    for (unsigned W : allowedWidths) {
      if (W >= N) {
        return W;
      }
    }
    return 0; // unreachable
  }

  bool needLegalization(const llvm::Instruction& I) {
    if (isIllegalInt(I.getType())) {
      return true;
    }
    for (const llvm::Value* U : I.operands()) {
      if (isIllegalInt(U->getType())) {
        return true;
      }
    }
    return false;
  }

  bool canPromoteOperandsInPlace(const llvm::Instruction& I) {
    // icmp eq/ne/unsigned, trunc->legal, uitofp, inttoptr
    if (I.getOpcode() == llvm::Instruction::ICmp) {
      auto pred = llvm::cast<llvm::ICmpInst>(I).getPredicate();
      return pred == llvm::CmpInst::ICMP_EQ || pred == llvm::CmpInst::ICMP_NE ||
             pred == llvm::CmpInst::ICMP_ULT || pred == llvm::CmpInst::ICMP_ULE ||
             pred == llvm::CmpInst::ICMP_UGT || pred == llvm::CmpInst::ICMP_UGE;
    } else if (I.getOpcode() == llvm::Instruction::Trunc ||
               I.getOpcode() == llvm::Instruction::UIToFP ||
               I.getOpcode() == llvm::Instruction::IntToPtr) {
      return true;
    }
    return false;
  }

  llvm::Value * sextInReg(llvm::IRBuilder<> &B, llvm::Value *V, unsigned From) {
    if (!V) {
      return nullptr;
    }
    unsigned W = V->getType()->getIntegerBitWidth();
    if (From == W) {
      return V;
    }
    return B.CreateAShr(B.CreateShl(V, W - From), W - From);
  }
} // namespace

llvm::PreservedAnalyses IntegerLegalizationPass::fail(llvm::Instruction* I) {
  llvm::SmallString<256> Str;
  llvm::raw_svector_ostream rso(Str);
  I->print(rso);
  ErrorMessage = "IntegerLegalizationPass: unsupported instruction: " + Str.str().str();
  return llvm::PreservedAnalyses::all();
}

llvm::Value* IntegerLegalizationPass::getPromoted(llvm::Value *V) {
  auto it = Promoted.find(V);
  if (it != Promoted.end()) {
    return it->second;
  }

  unsigned N = V->getType()->getIntegerBitWidth(); // old width
  unsigned W = promoteWidth(N); // new width
  auto* ResultTy = llvm::IntegerType::get(M->getContext(), W);
  if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return llvm::ConstantInt::get(ResultTy, CI->getValue().zext(W));
  } else if (llvm::isa<llvm::PoisonValue>(V)) {
    return llvm::PoisonValue::get(ResultTy);
  } else if (llvm::isa<llvm::UndefValue>(V)) {
    return llvm::ConstantInt::get(ResultTy, 0);
  }
  return nullptr;
}

llvm::Value* IntegerLegalizationPass::promoteResult(llvm::IRBuilder<> &B, llvm::Value* V) {
  // isIllegalInt = true for this instruction
  unsigned N = V->getType()->getIntegerBitWidth(); // old width
  unsigned W = promoteWidth(N); // new width
  auto* ResultTy = llvm::IntegerType::get(M->getContext(), W);

  if (auto* BI = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
    auto* LHS = getPromoted(BI->getOperand(0));
    auto* RHS = getPromoted(BI->getOperand(1));
    if (!LHS || !RHS) {
      return nullptr;
    }
    llvm::APInt Mask = llvm::APInt::getLowBitsSet(W, N);

    llvm::Value *New = nullptr;
    switch (BI->getOpcode()) {
      case llvm::Instruction::And:
      case llvm::Instruction::Or:
      case llvm::Instruction::Xor:
      case llvm::Instruction::UDiv:
      case llvm::Instruction::URem:
      case llvm::Instruction::LShr:
        New = B.CreateBinOp(BI->getOpcode(), LHS, RHS);
        break;

      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::Shl:
        New = B.CreateAnd(B.CreateBinOp(BI->getOpcode(), LHS, RHS), Mask);
        break;

      case llvm::Instruction::SDiv:
      case llvm::Instruction::SRem:
        New = B.CreateAnd(B.CreateBinOp(BI->getOpcode(), sextInReg(B, LHS, N), sextInReg(B, RHS, N)), Mask);
        break;

      case llvm::Instruction::AShr:
        New = B.CreateAnd(B.CreateAShr(sextInReg(B, LHS, N), RHS), Mask);
        break;
      default:
        // error
        break;
    }
    if (New) {
      New->takeName(BI);
    }

    return New;
  } else if (auto* Trunc = llvm::dyn_cast<llvm::TruncInst>(V)) {
    auto* Op = Trunc->getOperand(0);
    Op = isIllegalInt(Op->getType()) ? getPromoted(Op) : Op;
    if (!Op) {
      return nullptr;
    }
    if (Op->getType()->getIntegerBitWidth() > W) {
      Op = B.CreateTrunc(Op, ResultTy);
    }

    return B.CreateAnd(Op, llvm::APInt::getLowBitsSet(W, N));
  } else if (auto* ZExt = llvm::dyn_cast<llvm::ZExtInst>(V)) {
    auto* Op = ZExt->getOperand(0);
    Op = isIllegalInt(Op->getType()) ? getPromoted(Op) : Op;
    if (!Op) {
      return nullptr;
    }
    if (Op->getType()->getIntegerBitWidth() < W) {
      return B.CreateZExt(Op, ResultTy);
    }
    return Op;
  } else if (auto* SExt = llvm::dyn_cast<llvm::SExtInst>(V)) {
    auto* Op = SExt->getOperand(0);
    Op = isIllegalInt(Op->getType()) ? sextInReg(B, getPromoted(Op), SExt->getOperand(0)->getType()->getIntegerBitWidth()) : Op;
    if (!Op) {
      return nullptr;
    }
    if (Op->getType()->getIntegerBitWidth() < W) {
      Op = B.CreateSExt(Op, ResultTy);
    }
    return B.CreateAnd(Op, llvm::APInt::getLowBitsSet(W, N));
  } else if (auto* Select = llvm::dyn_cast<llvm::SelectInst>(V)) {
    auto* Cond = Select->getCondition();
    auto* TrueVal = getPromoted(Select->getTrueValue());
    auto* FalseVal = getPromoted(Select->getFalseValue());
    if (!TrueVal || !FalseVal) {
      return nullptr;
    }
    return B.CreateSelect(Cond, TrueVal, FalseVal);
  } else if (auto* Freeze = llvm::dyn_cast<llvm::FreezeInst>(V)) {
    auto* Op = getPromoted(Freeze->getOperand(0));
    if (!Op) {
      return nullptr;
    }
    return B.CreateAnd(B.CreateFreeze(Op), llvm::APInt::getLowBitsSet(W, N));
  }

  return nullptr;
}

bool IntegerLegalizationPass::rebuildLegal(llvm::IRBuilder<> &B, llvm::Instruction* I) {
  // isIllegalInt = false for this instruction, but at least one operand is illegal
  if (auto* ZExt = llvm::dyn_cast<llvm::ZExtInst>(I)) {
    auto* Op = getPromoted(ZExt->getOperand(0));
    if (!Op) {
      return false;
    }
    if (Op->getType()->getIntegerBitWidth() < ZExt->getType()->getIntegerBitWidth()) {
      Op = B.CreateZExt(Op, ZExt->getType());
    }
    I->replaceAllUsesWith(Op);
    return true;
  } else if (auto* SExt = llvm::dyn_cast<llvm::SExtInst>(I)) {
    auto* Op = sextInReg(B, getPromoted(SExt->getOperand(0)), SExt->getOperand(0)->getType()->getIntegerBitWidth());
    if (!Op) {
      return false;
    }
    if (Op->getType()->getIntegerBitWidth() < SExt->getType()->getIntegerBitWidth()) {
      Op = B.CreateSExt(Op, SExt->getType());
    }
    I->replaceAllUsesWith(Op);
    return true;
  } else if (auto* Cmp = llvm::dyn_cast<llvm::ICmpInst>(I)) {
    auto pred = Cmp->getPredicate();
    if (pred == llvm::CmpInst::ICMP_SLT || pred == llvm::CmpInst::ICMP_SLE ||
        pred == llvm::CmpInst::ICMP_SGT || pred == llvm::CmpInst::ICMP_SGE)
    {
      auto* LHS = sextInReg(B, getPromoted(Cmp->getOperand(0)), Cmp->getOperand(0)->getType()->getIntegerBitWidth());
      auto* RHS = sextInReg(B, getPromoted(Cmp->getOperand(1)), Cmp->getOperand(1)->getType()->getIntegerBitWidth());
      if (!LHS || !RHS) {
        return false;
      }
      I->replaceAllUsesWith(B.CreateICmp(pred, LHS, RHS));
      return true;
    }
  }
  return false;
}

llvm::PreservedAnalyses IntegerLegalizationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  bool Changed = false;
  this->M = &M;

  for (auto& F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    llvm::SmallVector<llvm::Instruction *> Worklist;
    llvm::ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

    for (auto* BB : RPOT) {
      for (auto& I : *BB) {
        if (needLegalization(I)) {
          if (isIllegalInt(I.getType()) && promoteWidth(I.getType()->getIntegerBitWidth()) == 0) {
            return fail(&I);
          }
          for (const llvm::Value* U : I.operands()) {
            if (isIllegalInt(U->getType()) && promoteWidth(U->getType()->getIntegerBitWidth()) == 0) {
              return fail(&I);
            }
          }
          Worklist.push_back(&I);
        }
      }
    }
    if (Worklist.empty()) {
      continue;
    }

    Changed = true;
    Promoted.clear();
    llvm::SmallPtrSet<llvm::Instruction *, 16> Kept;
    llvm::SmallVector<std::pair<llvm::PHINode *, llvm::PHINode *>> PHIs; // old PHI -> new PHI
    llvm::IRBuilder<> B(F.getContext());

    for (auto* I : Worklist) {
      B.SetInsertPoint(I);

      if (auto *P = llvm::dyn_cast<llvm::PHINode>(I)) {
        unsigned N = I->getType()->getIntegerBitWidth();
        unsigned W = promoteWidth(N);
        auto *NewP = llvm::PHINode::Create(llvm::IntegerType::get(M.getContext(), W), P->getNumIncomingValues(), "", P->getIterator());
        NewP->takeName(P);
        Promoted[P] = NewP;
        PHIs.push_back({P, NewP});
        continue;
      }

      auto* ResultTy = I->getType();
      if (isIllegalInt(ResultTy)) {
        auto* New = promoteResult(B, I);
        if (!New) {
          return fail(I);
        }
        Promoted[I] = New;
      } else if (canPromoteOperandsInPlace(*I)) {
        // rebuild with promoted operands
        for (unsigned i = 0; i < I->getNumOperands(); ++i) {
          auto* Op = I->getOperand(i);
          if (isIllegalInt(Op->getType())) {
            auto* NewOp = getPromoted(Op);
            if (!NewOp) {
              return fail(I);
            }
            I->setOperand(i, NewOp);
          }
        }
        Kept.insert(I);
      } else if (!rebuildLegal(B, I)) {
        return fail(I);
      }
    }

    for (auto [Old, New] : PHIs) {
      for (unsigned i = 0; i < Old->getNumIncomingValues(); ++i) {
        auto *Incoming = Old->getIncomingValue(i);
        if (isIllegalInt(Incoming->getType())) {
          Incoming = getPromoted(Incoming);
        }
        if (!Incoming) {
          return fail(Old);
        }
        New->addIncoming(Incoming, Old->getIncomingBlock(i));
      }
    }

    for (llvm::Instruction *I : Worklist) {
      if (Kept.find(I) == Kept.end()) {
        I->dropAllReferences();
      }
    }
    for (llvm::Instruction *I : Worklist) {
      if (Kept.find(I) == Kept.end()) {
        I->eraseFromParent();
      }
    }
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
