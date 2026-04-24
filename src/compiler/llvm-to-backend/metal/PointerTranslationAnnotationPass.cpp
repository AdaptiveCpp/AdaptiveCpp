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

#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationAnnotationPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <llvm/ADT/SmallPtrSet.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

namespace {

AllocaInst *getBaseAlloca(Value *V) {
  while (V) {
    if (auto *AI = dyn_cast<AllocaInst>(V)) {
      return AI;
    }
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      V = BC->getOperand(0);
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
      V = ASC->getOperand(0);
      continue;
    }
    break;
  }
  return nullptr;
}

// AddressSpaceInferencePass can leave global-origin pointers in generic AS=0.
// Trace through PHIs, GEPs, casts, loads and tainted private allocas to recover
// whether a load/store location is logically global memory.
bool tracesToGlobal(
  Value *V, unsigned GlobalAS,
  const SmallPtrSetImpl<AllocaInst *> &TaintedAllocas,
  SmallPtrSetImpl<Value *> &Visited) {
  if (!V || !V->getType()->isPointerTy()) {
    return false;
  }

  unsigned AS = V->getType()->getPointerAddressSpace();
  if (AS == GlobalAS) {
    return true;
  }

  if (!Visited.insert(V).second) {
    return false;
  }

  if (AS == 5) {
    if (auto *AI = dyn_cast<AllocaInst>(V)) {
      return TaintedAllocas.count(AI) != 0;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      return tracesToGlobal(GEP->getPointerOperand(), GlobalAS, TaintedAllocas, Visited);
    }
    if (auto *BC = dyn_cast<BitCastInst>(V)) {
      return tracesToGlobal(BC->getOperand(0), GlobalAS, TaintedAllocas, Visited);
    }
    return false;
  }

  if (AS != 0) {
    return false;
  }

  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    return tracesToGlobal(ASC->getOperand(0), GlobalAS, TaintedAllocas, Visited);
  }
  if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
    return tracesToGlobal(GEP->getPointerOperand(), GlobalAS, TaintedAllocas, Visited);
  }
  if (auto *BC = dyn_cast<BitCastInst>(V)) {
    return tracesToGlobal(BC->getOperand(0), GlobalAS, TaintedAllocas, Visited);
  }
  if (auto *PHI = dyn_cast<PHINode>(V)) {
    for (Value *Inc : PHI->incoming_values()) {
      Type *T = Inc->getType();
      if (!T->isPointerTy()) {
        continue;
      }
      unsigned IncAS = T->getPointerAddressSpace();
      if (IncAS != 0 && IncAS != GlobalAS && IncAS != 5) {
        return false;
      }
    }
    for (Value *Inc : PHI->incoming_values()) {
      if (tracesToGlobal(Inc, GlobalAS, TaintedAllocas, Visited)) {
        return true;
      }
    }
    return false;
  }
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    return tracesToGlobal(LI->getPointerOperand(), GlobalAS, TaintedAllocas, Visited);
  }
  return false;
}

bool tracesToGlobal(Value *V, unsigned GlobalAS, const SmallPtrSetImpl<AllocaInst *> &TaintedAllocas) {
  SmallPtrSet<Value *, 16> Visited;
  return tracesToGlobal(V, GlobalAS, TaintedAllocas, Visited);
}

// Private pointer stacks copied from or filled with global-origin pointers need
// to be treated as global-origin sources themselves.
SmallPtrSet<AllocaInst *, 8> computeTaintedAllocas(Function &F, unsigned GlobalAS) {
  SmallPtrSet<AllocaInst *, 8> Tainted;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *MI = dyn_cast<MemCpyInst>(&I);
      if (!MI) {
        continue;
      }
      if (MI->getRawSource()->getType()->getPointerAddressSpace() != GlobalAS) {
        continue;
      }
      if (MI->getRawDest()->getType()->getPointerAddressSpace() != 5) {
        continue;
      }
      if (AllocaInst *AI = getBaseAlloca(MI->getRawDest())) {
        Tainted.insert(AI);
      }
    }
  }

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI) {
          continue;
        }
        if (SI->getPointerOperand()->getType()->getPointerAddressSpace() != 5) {
          continue;
        }
        Value *Val = SI->getValueOperand();
        if (!Val->getType()->isPointerTy()) {
          continue;
        }
        AllocaInst *AI = getBaseAlloca(SI->getPointerOperand());
        if (!AI || Tainted.count(AI)) {
          continue;
        }
        if (tracesToGlobal(Val, GlobalAS, Tainted)) {
          Tainted.insert(AI);
          Changed = true;
        }
      }
    }
  }
  return Tainted;
}

} // namespace

PointerTranslationAnnotationPass::PointerTranslationAnnotationPass(unsigned GlobalAS)
  : GlobalAS(GlobalAS)
{}

llvm::PreservedAnalyses PointerTranslationAnnotationPass::run(Module &M, ModuleAnalysisManager &MAM) {
  LLVMContext &Ctx = M.getContext();
  MDNode *EmptyMD = MDNode::get(Ctx, {});

  unsigned LoadsAnnotated = 0;
  unsigned StoresAnnotated = 0;

  for (Function &F : M) {
    auto TaintedAllocas = computeTaintedAllocas(F, GlobalAS);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (!LI->getType()->isPointerTy()) {
            continue;
          }
          if (!tracesToGlobal(LI->getPointerOperand(), GlobalAS, TaintedAllocas)) {
            continue;
          }

          LI->setMetadata(MetalPtrLoadNeedsTranslationMD, EmptyMD);
          ++LoadsAnnotated;
          HIPSYCL_DEBUG_INFO << "Metal PtrTranslation: annotated load in "
                             << F.getName() << ": " << *LI << "\n";
          continue;
        }

        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (!SI->getValueOperand()->getType()->isPointerTy()) {
            continue;
          }
          if (!tracesToGlobal(SI->getPointerOperand(), GlobalAS, TaintedAllocas)) {
            continue;
          }

          SI->setMetadata(MetalPtrStoreNeedsTranslationMD, EmptyMD);
          ++StoresAnnotated;
          HIPSYCL_DEBUG_INFO << "Metal PtrTranslation: annotated store in "
                             << F.getName() << ": " << *SI << "\n";
          continue;
        }
      }
    }
  }

  HIPSYCL_DEBUG_INFO << "Metal PointerTranslationAnnotationPass: annotated "
                     << LoadsAnnotated << " load(s) and "
                     << StoresAnnotated << " store(s) for address translation.\n";

  return (LoadsAnnotated + StoresAnnotated) > 0
    ? PreservedAnalyses::none()
    : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
