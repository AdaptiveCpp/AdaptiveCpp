#include "hipSYCL/compiler/LoopSplitter.hpp"

#include "hipSYCL/common/debug.hpp"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &ost, const llvm::StringRef &strRef) {
  return ost << strRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationInfo::AnalyzeModule(llvm::Module &module) {
  for (auto &I : module.globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get());
        auto *FUNC = llvm::dyn_cast<llvm::Function>(CS->getOperand(0)->getOperand(0));
        auto *AnnotationGL = llvm::dyn_cast<llvm::GlobalVariable>(CS->getOperand(1)->getOperand(0));
        llvm::StringRef annotation =
            llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())->getAsCString();
        if (annotation.compare(SplitterAnnotation) == 0) {
          splitterFuncs_.insert(FUNC);
          HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << FUNC->getName() << "\n";
        }
      }
    }
  }
  return false;
}

hipsycl::compiler::SplitterAnnotationInfo::SplitterAnnotationInfo(llvm::Module &module) { AnalyzeModule(module); }

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  if (splitterAnnotation_)
    return false;
  splitterAnnotation_ = SplitterAnnotationInfo{*F.getParent()};
  return false;
}

hipsycl::compiler::SplitterAnnotationAnalysis::Result
hipsycl::compiler::SplitterAnnotationAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  return SplitterAnnotationInfo{M};
}

namespace {

llvm::Loop *UpdateDTAndLI(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

bool InlineSplitterCallTree(llvm::CallBase *CI, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  if (CI->getCalledFunction()->isIntrinsic() || SAA.IsSplitterFunc(CI->getCalledFunction()))
    return false;

  // needed to be valid for success log
  const auto calleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR <= 10
  llvm::InlineResult ILR = llvm::InlineFunction(CI, IFI, nullptr);
  if (!static_cast<bool>(ILR)) {
    HIPSYCL_DEBUG_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.message << "'" << std::endl;
#else
  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI, nullptr);
  if (!ILR.isSuccess()) {
    HIPSYCL_DEBUG_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.getFailureReason() << "'"
                          << std::endl;
#endif
    return false;
  }

  HIPSYCL_DEBUG_INFO << "LoopSplitter inlined function <" << calleeName << ">" << std::endl;
  return true;
}

bool InlineCallsInBasicBlock(llvm::BasicBlock &BB, const llvm::SmallPtrSet<llvm::Function *, 8> &splitterCallers,
                             const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  bool changed = false;
  bool lastChanged = false;

  do {
    lastChanged = false;
    for (auto &I : BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() && splitterCallers.find(callI->getCalledFunction()) != splitterCallers.end()) {
          lastChanged = InlineSplitterCallTree(callI, SAA);
          if (lastChanged)
            break;
        }
      }
    }
    if (lastChanged)
      changed = true;
  } while (lastChanged);

  return changed;
}

//! \pre all contained functions are non recursive!
// todo: have a recursive-ness termination
bool InlineCallsInLoop(llvm::Loop *&L, const llvm::SmallPtrSet<llvm::Function *, 8> &splitterCallers,
                       const hipsycl::compiler::SplitterAnnotationInfo &SAA, llvm::LoopInfo &LI,
                       llvm::DominatorTree &DT) {
  bool changed = false;
  bool lastChanged = false;

  llvm::BasicBlock *B = L->getBlocks()[0];
  llvm::Function &F = *B->getParent();

  do {
    lastChanged = false;
    for (auto *BB : L->getBlocks()) {
      lastChanged = InlineCallsInBasicBlock(*BB, splitterCallers, SAA);
      if (lastChanged)
        break;
    }
    if (lastChanged) {
      changed = true;
      L = UpdateDTAndLI(LI, DT, B, F);
    }
  } while (lastChanged);

  return changed;
}

//! \pre \a F is not recursive!
// todo: have a recursive-ness termination
bool FillTransitiveSplitterCallers(llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter) {
  if (F.isDeclaration() && !F.isIntrinsic()) {
    HIPSYCL_DEBUG_WARNING << F.getName() << " is not defined!" << std::endl;
  }
  if (SAA.IsSplitterFunc(&F)) {
    FuncsWSplitter.insert(&F);
    return true;
  } else if (FuncsWSplitter.find(&F) != FuncsWSplitter.end())
    return true;

  bool found = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() &&
            FillTransitiveSplitterCallers(*callI->getCalledFunction(), SAA, FuncsWSplitter)) {
          FuncsWSplitter.insert(&F);
          found = true;
        }
      }
    }
  }
  return found;
}

bool FillTransitiveSplitterCallers(llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter) {
  bool found = false;
  for (auto *BB : L.getBlocks()) {
    for (auto &I : *BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() &&
            FillTransitiveSplitterCallers(*callI->getCalledFunction(), SAA, FuncsWSplitter))
          found = true;
      }
    }
  }
  return found;
}

void FindAllSplitterCalls(const llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                          llvm::SmallVector<llvm::CallBase *, 8> &barriers) {
  for (auto *BB : L.getBlocks()) {
    for (auto &I : *BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() && SAA.IsSplitterFunc(callI->getCalledFunction())) {
          barriers.push_back(callI);
        }
      }
    }
  }
}
bool IsInConditional(llvm::CallBase *barrierI, llvm::DominatorTree &dt, llvm::BasicBlock *latch) {
  return !dt.properlyDominates(barrierI->getParent(), latch);
}

bool FillDescendantsExcl(const llvm::BasicBlock *root, const llvm::BasicBlock *excl, llvm::DominatorTree &DT,
                         llvm::SmallVectorImpl<llvm::BasicBlock *> &searchBlocks) {
  auto *rootNode = DT.getNode(root);
  if (!rootNode)
    return false;

  llvm::SmallVector<const llvm::DomTreeNodeBase<llvm::BasicBlock> *, 8> WL;
  WL.append(rootNode->begin(), rootNode->end());

  while (!WL.empty()) {
    const llvm::DomTreeNodeBase<llvm::BasicBlock> *N = WL.pop_back_val();
    if (N->getBlock() != excl) {
      searchBlocks.push_back(N->getBlock());
      WL.append(N->begin(), N->end());
    }
  }
  return !searchBlocks.empty();
}

void InsertOperands(llvm::SmallPtrSet<llvm::Value *, 8> &WL, const llvm::SmallPtrSet<llvm::Value *, 8> &DL,
                    const llvm::SmallVectorImpl<llvm::BasicBlock *> &searchBlocks,
                    llvm::SmallPtrSetImpl<llvm::Instruction *> &result, const llvm::Instruction *I) {
  auto argsRange = I->operands();
  if (auto *cI = llvm::dyn_cast<llvm::CallBase>(I)) {
    argsRange = cI->args();
  }

  //  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "adding operands: ";
  //  I->print(llvm::outs());
  //  llvm::outs() << "\n";

  for (auto &OP : argsRange) {
    if (/*OP->getType()->isPointerTy() && */ llvm::isa<llvm::Instruction>(OP) && DL.find(OP) == DL.end()) {
      if (!WL.insert(OP).second)
        continue;
      OP->print(llvm::outs());
      //      llvm::outs() << " as op: ";
      //      OP->printAsOperand(llvm::outs());
      //      llvm::outs() << "\n";
      if (auto *Inst = llvm::cast<llvm::Instruction>(static_cast<llvm::Value *>(OP))) {
        if (std::find(searchBlocks.begin(), searchBlocks.end(), Inst->getParent()) != searchBlocks.end() &&
            !llvm::isa<llvm::BranchInst>(OP) && !llvm::isa<llvm::StoreInst>(I)) {
          //          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "........ to result";
          if (result.insert(Inst).second) {
            //            llvm::outs() << " !!!!! NEW";
          }
          //          llvm::outs() << "\n";
        }
        InsertOperands(WL, DL, searchBlocks, result, Inst);
      }
    }
  }
}

bool BuildTransitiveDependencyHullFromWL(const llvm::SmallVectorImpl<llvm::BasicBlock *> &searchBlocks,
                                         llvm::SmallPtrSetImpl<llvm::Instruction *> &result,
                                         llvm::SmallPtrSet<llvm::Value *, 8> &WL,
                                         llvm::SmallPtrSet<llvm::Value *, 8> &DL) {
  while (!WL.empty()) {
    auto *V = *WL.begin();
    WL.erase(V);
    DL.insert(V);

    for (auto &U : V->uses()) {
      //      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO;
      //      U.getUser()->print(llvm::outs());
      //      llvm::outs() << "\n";

      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
        bool toWL = true;
        if (std::find(searchBlocks.begin(), searchBlocks.end(), I->getParent()) != searchBlocks.end() &&
            !llvm::isa<llvm::BranchInst>(I) && !llvm::isa<llvm::StoreInst>(I)) {
          //          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "........ to result";
          toWL = result.insert(I).second;
          //          if (toWL)
          //            llvm::outs() << " !!!!! NEW";
          //          llvm::outs() << "\n";
        }

        if (DL.find(I) == DL.end()) {
          if (toWL && !I->user_empty()) {
            //            llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "--------- to worklist"
            //                         << "\n";

            if (!WL.insert(I).second)
              continue;
          }
          InsertOperands(WL, DL, searchBlocks, result, I);
        }
      }
    }
  }
  return !result.empty();
}

bool FillDependingInsts(const llvm::SmallVectorImpl<llvm::BasicBlock *> &baseBlocks,
                        const llvm::SmallVectorImpl<llvm::BasicBlock *> &searchBlocks,
                        llvm::SmallPtrSetImpl<llvm::Instruction *> &result) {
  llvm::SmallPtrSet<llvm::Value *, 8> WL;
  llvm::SmallPtrSet<llvm::Value *, 8> DL;
  for (auto *BB : baseBlocks) {
    for (auto &V : *BB) {
      WL.insert(&V);
      InsertOperands(WL, DL, searchBlocks, result, &V);
    }
  }

  return BuildTransitiveDependencyHullFromWL(searchBlocks, result, WL, DL);
}

bool FillDependingInsts(const llvm::SmallPtrSetImpl<llvm::Instruction *> &startList,
                        const llvm::SmallVectorImpl<llvm::BasicBlock *> &searchBlocks,
                        llvm::SmallPtrSetImpl<llvm::Instruction *> &result) {
  llvm::SmallPtrSet<llvm::Value *, 8> WL;
  llvm::SmallPtrSet<llvm::Value *, 8> DL;
  for (auto *I : startList) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO;
    I->print(llvm::outs());
    llvm::outs() << "\n";
    if (I->hasValueHandle())
      WL.insert(I);
    InsertOperands(WL, DL, searchBlocks, result, I);
  }

  return BuildTransitiveDependencyHullFromWL(searchBlocks, result, WL, DL);
}

// void FilterOverwrittenMemAccesses(llvm::SmallVectorImpl<llvm::Instruction*> &insts, llvm::MemoryDependenceResults&
// memDep)
//{
//  for(auto it = insts.rbegin(); it != insts.rend(); ++it)
//  {
//    if(auto *li = llvm::dyn_cast<llvm::LoadInst>(*it))
//    {
//      llvm::errs() << HIPSYCL_DEBUG_LEVEL_INFO;
//      li->print(llvm::errs());
//      llvm::errs() << '\n';
//      memDep.getDependencyFrom()
//    }
//    else if(auto *si = llvm::dyn_cast<llvm::StoreInst>(*it))
//    {
//      llvm::errs() << HIPSYCL_DEBUG_LEVEL_INFO;
//      li->print(llvm::errs());
//      llvm::errs() << '\n';
//    }
//  }
//}

/*!
 * In _too simple_ loops, we might not have a dedicated latch.. so make one!
 * Only simple / canonical loops supported.
 *
 * @param L The loop without a dedicated latch.
 * @param BodyBlock The loop body.
 * @return The new latch block, if possible containing the loop induction instruction.
 */
llvm::BasicBlock *MakeLatch(const llvm::Loop *L, llvm::BasicBlock *BodyBlock, llvm::LoopInfo &LI,
                            llvm::DominatorTree &DT) {
  llvm::BasicBlock *Latch =
      llvm::SplitBlock(BodyBlock, BodyBlock->getTerminator(), &DT, &LI, nullptr, BodyBlock->getName() + ".latch");

  assert(L->getCanonicalInductionVariable() && "must be canonical loop!");
  llvm::Value *InductionValue = L->getCanonicalInductionVariable()->getIncomingValueForBlock(Latch);
  if (auto *InductionInstr = llvm::dyn_cast<llvm::Instruction>(InductionValue)) {
    auto *NewIndInstr = InductionInstr->clone();
    NewIndInstr->insertBefore(Latch->getFirstNonPHI());
    InductionInstr->replaceAllUsesWith(NewIndInstr);
    InductionInstr->eraseFromParent();
  } else {
    llvm::errs() << HIPSYCL_DEBUG_PREFIX_ERROR << "Induction variable must be an instruction!\n";
  }
  return Latch;
}

bool splitLoop(llvm::Loop *L, llvm::LoopInfo &LI, const std::function<void(llvm::Loop &)> &LoopAdder,
               const llvm::LoopAccessInfo &LAI, llvm::DominatorTree &DT, llvm::ScalarEvolution &SE,
               const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  if (!(*L->block_begin())->getParent()->getName().startswith(".omp_outlined")) {
    // are we in kernel?
    return false;
  }

  if (L->getLoopDepth() != 2) {
    // only second-level loop have to be considered as work-item loops -> must be using collapse on multi-dim kernels
    HIPSYCL_DEBUG_INFO << "Not work-item loop!" << L << std::endl;
    return false;
  }

  llvm::Function *F = L->getBlocks()[0]->getParent();

  llvm::AssumptionCache AC(*F);

  llvm::SmallPtrSet<llvm::Function *, 8> splitterCallers;
  if (!FillTransitiveSplitterCallers(*L, SAA, splitterCallers)) {
    HIPSYCL_DEBUG_INFO << "Transitively no splitter found." << L << std::endl;
    return false;
  }

  bool changed = InlineCallsInLoop(L, splitterCallers, SAA, LI, DT);

  llvm::SmallVector<llvm::CallBase *, 8> barriers;
  FindAllSplitterCalls(*L, SAA, barriers);

  if (barriers.empty()) {
    HIPSYCL_DEBUG_INFO << "No splitter found." << std::endl;
    return changed;
  }

  std::size_t bC = 0;
  F->print(llvm::outs());
  for (auto *barrier : barriers) {
    changed = true;
    ++bC;

    HIPSYCL_DEBUG_INFO << "Found splitter at " << barrier->getCalledFunction()->getName() << std::endl;

    HIPSYCL_DEBUG_INFO << "Found header: " << L->getHeader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found pre-header: " << L->getLoopPreheader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found exit block: " << L->getExitBlock() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found latch block: " << L->getLoopLatch() << std::endl;

    auto *oldBlock = barrier->getParent();
    if (LI.getLoopFor(oldBlock) != L) {
      HIPSYCL_DEBUG_ERROR << "Barrier must be directly in item loop for now." << std::endl;
      continue;
    }

    llvm::Loop *parentLoop = L->getParentLoop();
    llvm::BasicBlock *header = L->getHeader();
    llvm::BasicBlock *preHeader = L->getLoopPreheader();
    llvm::BasicBlock *exitBlock = L->getExitBlock();
    llvm::BasicBlock *latch = L->getLoopLatch();

    if (latch == oldBlock) {
      latch = MakeLatch(L, oldBlock, LI, DT);
    }

    if (IsInConditional(barrier, DT, latch)) {
      HIPSYCL_DEBUG_INFO << "is in conditional" << std::endl;
    }

    llvm::SmallPtrSet<llvm::Instruction *, 8> dependingInsts;
    llvm::SmallVector<llvm::BasicBlock *, 2> baseBlocks = {header};
    llvm::SmallVector<llvm::BasicBlock *, 8> searchBlocks;
    //    DT.getDescendants(header, searchBlocks);
    FillDescendantsExcl(header, latch, DT, searchBlocks);
    //    for (auto *BB : searchBlocks) {
    //      BB->print(llvm::errs());
    //    }
    if (FillDependingInsts(baseBlocks, searchBlocks, dependingInsts)) {
      HIPSYCL_DEBUG_INFO << "has dependencies from header" << std::endl;

      for (auto *I : dependingInsts) {
        I->print(llvm::outs());
        llvm::outs() << "\n";
      }
    } else {
      HIPSYCL_DEBUG_WARNING << "what's wrong with you.. empty work item?!" << std::endl;
    }

    llvm::Loop &newLoop = *LI.AllocateLoop();
    parentLoop->addChildLoop(&newLoop);

    llvm::ValueToValueMapTy vMap;
    vMap[preHeader] = header;

    const std::string blockNameSuffix = "split" + std::to_string(bC);
    llvm::ClonedCodeInfo clonedCodeInfo;
    auto *newHeader = llvm::CloneBasicBlock(header, vMap, blockNameSuffix, F, &clonedCodeInfo, nullptr);
    vMap[header] = newHeader;
    newLoop.addBlockEntry(newHeader);
    newLoop.moveToHeader(newHeader);

    auto *newLatch = llvm::CloneBasicBlock(latch, vMap, blockNameSuffix, F, &clonedCodeInfo, nullptr);
    vMap[latch] = newLatch;
    newLoop.addBlockEntry(newLatch);

    auto *newBlock = llvm::SplitBlock(oldBlock, barrier, &DT, &LI, nullptr, oldBlock->getName() + blockNameSuffix);
    L->removeBlockFromLoop(newBlock);
    newLoop.addBlockEntry(newBlock);
    barrier->eraseFromParent();
    dependingInsts.erase(barrier);

    // connect new loop
    newHeader->getTerminator()->setSuccessor(0, newBlock);
    newHeader->getTerminator()->setSuccessor(1, exitBlock);
    DT.addNewBlock(newHeader, header);
    DT.changeImmediateDominator(newBlock, newHeader);
    DT.changeImmediateDominator(exitBlock, newHeader);

    llvm::SmallVector<llvm::BasicBlock *, 2> preds{llvm::pred_begin(latch), llvm::pred_end(latch)};
    llvm::BasicBlock *ncd = nullptr;
    for (auto *pred : preds) {
      std::size_t succIdx = 0;
      for (auto *succ : llvm::successors(pred)) {
        if (succ == latch)
          break;
        ++succIdx;
      }
      ncd = ncd ? DT.findNearestCommonDominator(ncd, pred) : pred;

      pred->getTerminator()->setSuccessor(succIdx, newLatch);
    }
    DT.addNewBlock(newLatch, ncd);

    newLatch->getTerminator()->setSuccessor(0, newHeader);
    //    DT.changeImmediateDominator(newHeader, newLatch);

    // fix old loop
    header->getTerminator()->setSuccessor(1, newHeader);
    oldBlock->getTerminator()->setSuccessor(0, latch);
    DT.changeImmediateDominator(latch, oldBlock);
    //    DT.changeImmediateDominator(header, newHeader);

    for (auto *subLoop : L->getSubLoops()) {
      auto *newParent = LI.getLoopFor(subLoop->getLoopPreheader());
      if (newParent != L) {
        HIPSYCL_DEBUG_INFO << "new parent for subloop: " << newParent << std::endl;
        L->removeChildLoop(subLoop);
        newParent->addChildLoop(subLoop);
      }
    }

    llvm::SmallVector<llvm::BasicBlock *, 8> bbToRemap;
    DT.getDescendants(newHeader, bbToRemap);
    HIPSYCL_DEBUG_INFO << "BLOCKS TO REMAP " << bbToRemap.size();
    llvm::SmallVector<llvm::BasicBlock *, 8> sBs;
    FillDescendantsExcl(header, newHeader, DT, sBs);
    sBs.erase(std::find(sBs.begin(), sBs.end(), latch));

    llvm::SmallPtrSet<llvm::Instruction *, 8> instsToCopy;
    llvm::SmallPtrSet<llvm::Instruction *, 8> instsInNew;
    for (auto *I : dependingInsts) {
      if (std::find(bbToRemap.begin(), bbToRemap.end(), I->getParent()) != bbToRemap.end())
        instsInNew.insert(I);
    }
    FillDependingInsts(instsInNew, sBs, instsToCopy);
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "insts to copy: "
                 << "\n";
    llvm::SmallVector<llvm::Instruction *, 8> instsToCopySorted{instsToCopy.begin(), instsToCopy.end()};
    std::sort(instsToCopySorted.begin(), instsToCopySorted.end(),
              [&](auto LHS, auto RHS) { return DT.dominates(LHS, RHS); });
    //    MemDepP.releaseMemory();
    //    MemDepP.getMemDep();
    llvm::Instruction *insPt = &*newBlock->getFirstInsertionPt();
    for (auto *I : instsToCopySorted) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO;
      I->print(llvm::outs());
      llvm::outs() << "\n";
      auto *clonedI = I->clone();
      vMap[I] = clonedI;
      //      if(insPt)
      //        clonedI->insertAfter(insPt);
      //      else
      clonedI->insertBefore(insPt);
      //      insPt = clonedI;
    }

    llvm::remapInstructionsInBlocks(bbToRemap, vMap);

    for (auto *block : L->getParentLoop()->blocks()) {
      if (!block->getParent())
        block->print(llvm::errs());
    }
    HIPSYCL_DEBUG_INFO << "new loopx.. " << &newLoop << " with parent " << newLoop.getParentLoop() << std::endl;
    DT.print(llvm::errs());
    L = UpdateDTAndLI(LI, DT, newLatch, *L->getHeader()->getParent());
    DT.print(llvm::errs());
    LoopAdder(*L);

    HIPSYCL_DEBUG_INFO << "new loop.. " << L << " with parent " << L->getParentLoop() << std::endl;

    //    HIPSYCL_DEBUG_INFO << "new exit block: " << LIW.getLoopInfo().getLoopFor(newBlock)->getExitBlock() <<
    //    std::endl; HIPSYCL_DEBUG_INFO << "old exit block: " << L->getExitBlock() << std::endl;

    for (auto *block : L->getParentLoop()->blocks()) {
      llvm::SimplifyInstructionsInBlock(block);
    }

    llvm::simplifyLoop(L->getParentLoop(), &DT, &LI, &SE, &AC, nullptr, false);
    F->viewCFG();
    //    DTW.getDomTree().viewGraph();

    if (llvm::verifyFunction(*F, &llvm::errs())) {
      HIPSYCL_DEBUG_ERROR << "function verification failed" << std::endl;
    }
  }
  F->print(llvm::outs());
  return changed;
}

} // namespace

bool hipsycl::compiler::LoopSplitAtBarrierPassLegacy::runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) {
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<llvm::AAResultsWrapperPass>();

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &SE = getAnalysis<llvm::ScalarEvolutionWrapperPass>().getSE();
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  llvm::LoopAccessInfo LAI(L, &SE, nullptr, &AA.getAAResults(), &DT, &LI);
  return splitLoop(
      L, LI, [&LPM](llvm::Loop &L) { LPM.addLoop(L); }, LAI, DT, SE, SAA);
}

void hipsycl::compiler::LoopSplitAtBarrierPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::AAResultsWrapperPass>();
  AU.addPreserved<llvm::AAResultsWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses hipsycl::compiler::LoopSplitAtBarrierPass::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                                                                       llvm::LoopStandardAnalysisResults &AR,
                                                                       llvm::LPMUpdater &LPMU) {
  auto &SAA = AM.getResult<SplitterAnnotationAnalysis>(L, AR);
  const auto &LAI = AM.getResult<llvm::LoopAccessAnalysis>(L, AR);
  if (!splitLoop(
          &L, AR.LI, [&LPMU](llvm::Loop &L) { LPMU.addSiblingLoops({&L}); }, LAI, AR.DT, AR.SE, SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA = llvm::getLoopPassPreservedAnalyses();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  PA.preserve<llvm::AAManager>();
  return PA;
}

char hipsycl::compiler::SplitterAnnotationAnalysisLegacy::ID = 0;
char hipsycl::compiler::LoopSplitAtBarrierPassLegacy::ID = 0;
llvm::AnalysisKey hipsycl::compiler::SplitterAnnotationAnalysis::Key;
