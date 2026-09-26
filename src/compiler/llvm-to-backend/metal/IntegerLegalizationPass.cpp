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
} // namespace

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
        break;

      case llvm::Instruction::AShr:
        New = B.CreateAnd(B.CreateAShr(sextInReg(LHS), RHS), Mask);
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
  }

  return nullptr;
}

llvm::PreservedAnalyses IntegerLegalizationPass::fail(llvm::Instruction* I) {
  llvm::SmallString<256> Str;
  llvm::raw_svector_ostream rso(Str);
  I->print(rso);
  ErrorMessage = "Unsupported instruction: " + Str.str().str();
  return llvm::PreservedAnalyses::all();
}

bool IntegerLegalizationPass::rebuildLegal(llvm::IRBuilder<> &B, llvm::Instruction* I) {
  return false;
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
  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
