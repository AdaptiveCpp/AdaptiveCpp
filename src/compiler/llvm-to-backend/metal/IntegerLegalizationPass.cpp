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

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>

#include <set>

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
} // namespace

llvm::Value * IntegerLegalizationPass::getPromoted(llvm::Value *V) {
  if (Promoted.count(V)) {
    return Promoted[V];
  }
  auto* ResultTy = V->getType();
  if (!isIllegalInt(ResultTy)) {
    return nullptr;
  }
  ResultTy = llvm::IntegerType::get(M->getContext(), promoteWidth(ResultTy->getIntegerBitWidth()));
  unsigned N = V->getType()->getIntegerBitWidth(); // old width
  unsigned W = ResultTy->getIntegerBitWidth(); // new width
  if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    auto *NewCI = llvm::ConstantInt::get(ResultTy, CI->getValue().zext(W));
    return Promoted[CI] = NewCI;
  } else if (auto* Undef = llvm::dyn_cast<llvm::UndefValue>(V)) {
    auto *NewUndef = llvm::ConstantInt::get(ResultTy, 0);
    return Promoted[Undef] = NewUndef;
  } else if (auto* Poison = llvm::dyn_cast<llvm::PoisonValue>(V)) {
    auto *NewPoison = llvm::PoisonValue::get(ResultTy);
    return Promoted[Poison] = NewPoison;
  } else if (auto* BI = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
    auto* LHS = getPromoted(BI->getOperand(0));
    auto* RHS = getPromoted(BI->getOperand(1));
    llvm::APInt Mask = llvm::APInt::getLowBitsSet(W, N);

    llvm::IRBuilder<> B(BI);

    auto sextInReg = [&](llvm::Value *V) {
      return B.CreateAShr(B.CreateShl(V, W - N), W - N);
    };

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
        New = B.CreateAnd(B.CreateBinOp(BI->getOpcode(), sextInReg(LHS), sextInReg(RHS)), Mask);

      case llvm::Instruction::AShr:
        New = B.CreateAnd(B.CreateAShr(sextInReg(LHS), RHS), Mask);
        break;
      default:
        // error
        break;
    }
    if (New) {
      New->takeName(BI);
      Promoted[BI] = New;
    }
  }

  return nullptr;
}

llvm::PreservedAnalyses IntegerLegalizationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
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
          Worklist.push_back(&I);
        }
      }
    }
    if (Worklist.empty()) {
      continue;
    }

    Promoted.clear();
    llvm::SmallVector<std::pair<llvm::PHINode *, llvm::PHINode *>> PHIs; // old PHI -> new PHI

    for (auto* I : Worklist) {
      if (auto *P = llvm::dyn_cast<llvm::PHINode>(I)) {
        auto *NewP = llvm::PHINode::Create(llvm::IntegerType::get(M.getContext(), promoteWidth(P->getType()->getIntegerBitWidth())), P->getNumIncomingValues(), "", P->getIterator());
        NewP->takeName(P);
        Promoted[P] = NewP;
        PHIs.push_back({P, NewP});
        continue;
      }

      auto* ResultTy = I->getType();
      if (isIllegalInt(ResultTy)) {
        getPromoted(I);
      } else {
        // rebuild with promoted operands
        for (unsigned i = 0; i < I->getNumOperands(); ++i) {
          auto* Op = I->getOperand(i);
          if (isIllegalInt(Op->getType())) {
            auto* NewOp = getPromoted(Op);
            if (NewOp) {
              I->setOperand(i, NewOp);
            }
          }
        }
      }
    }

    for (auto [Old, New] : PHIs) {
      for (unsigned i = 0; i < Old->getNumIncomingValues(); ++i) {
        New->addIncoming(getPromoted(Old->getIncomingValue(i)), Old->getIncomingBlock(i));
      }
    }

    for (llvm::Instruction *I : Worklist) {
      I->dropAllReferences();
    }
    for (llvm::Instruction *I : Worklist) {
      I->eraseFromParent();
    }
  }
  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
