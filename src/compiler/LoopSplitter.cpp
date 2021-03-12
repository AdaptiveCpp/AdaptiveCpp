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

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &Ost, const llvm::StringRef &StrRef) {
  return Ost << StrRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationInfo::analyzeModule(llvm::Module &Module) {
  for (auto &I : Module.globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto *OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get());
        auto *FUNC = llvm::dyn_cast<llvm::Function>(CS->getOperand(0)->getOperand(0));
        auto *AnnotationGL = llvm::dyn_cast<llvm::GlobalVariable>(CS->getOperand(1)->getOperand(0));
        llvm::StringRef Annotation =
            llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())->getAsCString();
        if (Annotation.compare(SplitterAnnotation) == 0) {
          SplitterFuncs.insert(FUNC);
          HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << FUNC->getName() << "\n";
        }
      }
    }
  }
  return false;
}

hipsycl::compiler::SplitterAnnotationInfo::SplitterAnnotationInfo(llvm::Module &Module) { analyzeModule(Module); }

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  if (SplitterAnnotation)
    return false;
  SplitterAnnotation = SplitterAnnotationInfo{*F.getParent()};
  return false;
}

hipsycl::compiler::SplitterAnnotationAnalysis::Result
hipsycl::compiler::SplitterAnnotationAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  return SplitterAnnotationInfo{M};
}

namespace {

llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

bool inlineSplitterCallTree(llvm::CallBase *CI, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  if (CI->getCalledFunction()->isIntrinsic() || SAA.isSplitterFunc(CI->getCalledFunction()))
    return false;

  // needed to be valid for success log
  const auto CalleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR <= 10
  llvm::InlineResult ILR = llvm::InlineFunction(CI, IFI, nullptr);
  if (!static_cast<bool>(ILR)) {
    HIPSYCL_DEBUG_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.message << "'" << std::endl;
#else
  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI, nullptr);
  if (!ILR.isSuccess()) {
    HIPSYCL_DEBUG_WARNING << "Failed to inline function <" << CalleeName << ">: '" << ILR.getFailureReason() << "'"
                          << std::endl;
#endif
    return false;
  }

  HIPSYCL_DEBUG_INFO << "LoopSplitter inlined function <" << CalleeName << ">" << std::endl;
  return true;
}

bool inlineCallsInBasicBlock(llvm::BasicBlock &BB, const llvm::SmallPtrSet<llvm::Function *, 8> &SplitterCallers,
                             const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  bool LastChanged = false;

  do {
    LastChanged = false;
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction() && SplitterCallers.find(CallI->getCalledFunction()) != SplitterCallers.end()) {
          LastChanged = inlineSplitterCallTree(CallI, SAA);
          if (LastChanged)
            break;
        }
      }
    }
    if (LastChanged)
      Changed = true;
  } while (LastChanged);

  return Changed;
}

//! \pre all contained functions are non recursive!
// todo: have a recursive-ness termination
bool inlineCallsInLoop(llvm::Loop *&L, const llvm::SmallPtrSet<llvm::Function *, 8> &SplitterCallers,
                       const hipsycl::compiler::SplitterAnnotationInfo &SAA, llvm::LoopInfo &LI,
                       llvm::DominatorTree &DT) {
  bool Changed = false;
  bool LastChanged = false;

  llvm::BasicBlock *B = L->getBlocks()[0];
  llvm::Function &F = *B->getParent();

  do {
    LastChanged = false;
    for (auto *BB : L->getBlocks()) {
      LastChanged = inlineCallsInBasicBlock(*BB, SplitterCallers, SAA);
      if (LastChanged)
        break;
    }
    if (LastChanged) {
      Changed = true;
      L = updateDtAndLi(LI, DT, B, F);
    }
  } while (LastChanged);

  return Changed;
}

//! \pre \a F is not recursive!
// todo: have a recursive-ness termination
bool fillTransitiveSplitterCallers(llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter) {
  if (F.isDeclaration() && !F.isIntrinsic()) {
    HIPSYCL_DEBUG_WARNING << F.getName() << " is not defined!" << std::endl;
  }
  if (SAA.isSplitterFunc(&F)) {
    FuncsWSplitter.insert(&F);
    return true;
  } else if (FuncsWSplitter.find(&F) != FuncsWSplitter.end())
    return true;

  bool Found = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction() &&
            fillTransitiveSplitterCallers(*CallI->getCalledFunction(), SAA, FuncsWSplitter)) {
          FuncsWSplitter.insert(&F);
          Found = true;
        }
      }
    }
  }
  return Found;
}

bool fillTransitiveSplitterCallers(llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter) {
  bool Found = false;
  for (auto *BB : L.getBlocks()) {
    for (auto &I : *BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction() &&
            fillTransitiveSplitterCallers(*CallI->getCalledFunction(), SAA, FuncsWSplitter))
          Found = true;
      }
    }
  }
  return Found;
}

void findAllSplitterCalls(const llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                          llvm::SmallVector<llvm::CallBase *, 8> &Barriers) {
  for (auto *BB : L.getBlocks()) {
    for (auto &I : *BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction() && SAA.isSplitterFunc(CallI->getCalledFunction())) {
          Barriers.push_back(CallI);
        }
      }
    }
  }
}
bool isInConditional(llvm::CallBase *BarrierI, llvm::DominatorTree &DT, llvm::BasicBlock *Latch) {
  return !DT.properlyDominates(BarrierI->getParent(), Latch);
}

bool fillDescendantsExcl(const llvm::BasicBlock *Root, const llvm::BasicBlock *Excl, llvm::DominatorTree &DT,
                         llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks) {
  auto *RootNode = DT.getNode(Root);
  if (!RootNode)
    return false;

  llvm::SmallVector<const llvm::DomTreeNodeBase<llvm::BasicBlock> *, 8> WL;
  WL.append(RootNode->begin(), RootNode->end());

  while (!WL.empty()) {
    const llvm::DomTreeNodeBase<llvm::BasicBlock> *N = WL.pop_back_val();
    if (N->getBlock() != Excl) {
      SearchBlocks.push_back(N->getBlock());
      WL.append(N->begin(), N->end());
    }
  }
  return !SearchBlocks.empty();
}

void insertOperands(llvm::SmallPtrSet<llvm::Value *, 8> &WL, const llvm::SmallPtrSet<llvm::Value *, 8> &DL,
                    const llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks,
                    llvm::SmallPtrSetImpl<llvm::Instruction *> &Result, const llvm::Instruction *I) {
  auto ArgsRange = I->operands();
  if (auto *CI = llvm::dyn_cast<llvm::CallBase>(I)) {
    ArgsRange = CI->args();
  }

  //  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "adding operands: ";
  //  I->print(llvm::outs());
  //  llvm::outs() << "\n";

  for (auto &OP : ArgsRange) {
    if (/*OP->getType()->isPointerTy() && */ llvm::isa<llvm::Instruction>(OP) && DL.find(OP) == DL.end()) {
      if (!WL.insert(OP).second)
        continue;
      OP->print(llvm::outs());
      //      llvm::outs() << " as op: ";
      //      OP->printAsOperand(llvm::outs());
      //      llvm::outs() << "\n";
      if (auto *Inst = llvm::cast<llvm::Instruction>(static_cast<llvm::Value *>(OP))) {
        if (std::find(SearchBlocks.begin(), SearchBlocks.end(), Inst->getParent()) != SearchBlocks.end() &&
            !llvm::isa<llvm::BranchInst>(OP) && !llvm::isa<llvm::StoreInst>(I)) {
          //          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "........ to result";
          if (Result.insert(Inst).second) {
            //            llvm::outs() << " !!!!! NEW";
          }
          //          llvm::outs() << "\n";
        }
        insertOperands(WL, DL, SearchBlocks, Result, Inst);
      }
    }
  }
}

bool buildTransitiveDependencyHullFromWl(const llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks,
                                         llvm::SmallPtrSetImpl<llvm::Instruction *> &Result,
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
        bool ToWl = true;
        if (std::find(SearchBlocks.begin(), SearchBlocks.end(), I->getParent()) != SearchBlocks.end() &&
            !llvm::isa<llvm::BranchInst>(I) && !llvm::isa<llvm::StoreInst>(I)) {
          //          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "........ to result";
          ToWl = Result.insert(I).second;
          //          if (toWL)
          //            llvm::outs() << " !!!!! NEW";
          //          llvm::outs() << "\n";
        }

        if (DL.find(I) == DL.end()) {
          if (ToWl && !I->user_empty()) {
            //            llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "--------- to worklist"
            //                         << "\n";

            if (!WL.insert(I).second)
              continue;
          }
          insertOperands(WL, DL, SearchBlocks, Result, I);
        }
      }
    }
  }
  return !Result.empty();
}

bool fillDependingInsts(const llvm::SmallVectorImpl<llvm::BasicBlock *> &BaseBlocks,
                        const llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks,
                        llvm::SmallPtrSetImpl<llvm::Instruction *> &Result) {
  llvm::SmallPtrSet<llvm::Value *, 8> WL;
  llvm::SmallPtrSet<llvm::Value *, 8> DL;
  for (auto *BB : BaseBlocks) {
    for (auto &V : *BB) {
      WL.insert(&V);
      insertOperands(WL, DL, SearchBlocks, Result, &V);
    }
  }

  return buildTransitiveDependencyHullFromWl(SearchBlocks, Result, WL, DL);
}

bool fillDependingInsts(const llvm::SmallPtrSetImpl<llvm::Instruction *> &StartList,
                        const llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks,
                        llvm::SmallPtrSetImpl<llvm::Instruction *> &Result) {
  llvm::SmallPtrSet<llvm::Value *, 8> WL;
  llvm::SmallPtrSet<llvm::Value *, 8> DL;
  for (auto *I : StartList) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO;
    I->print(llvm::outs());
    llvm::outs() << "\n";
    if (I->hasValueHandle())
      WL.insert(I);
    insertOperands(WL, DL, SearchBlocks, Result, I);
  }

  return buildTransitiveDependencyHullFromWl(SearchBlocks, Result, WL, DL);
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
llvm::BasicBlock *makeLatch(const llvm::Loop *L, llvm::BasicBlock *BodyBlock, llvm::LoopInfo &LI,
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

  llvm::SmallPtrSet<llvm::Function *, 8> SplitterCallers;
  if (!fillTransitiveSplitterCallers(*L, SAA, SplitterCallers)) {
    HIPSYCL_DEBUG_INFO << "Transitively no splitter found." << L << std::endl;
    return false;
  }

  bool Changed = inlineCallsInLoop(L, SplitterCallers, SAA, LI, DT);

  llvm::SmallVector<llvm::CallBase *, 8> Barriers;
  findAllSplitterCalls(*L, SAA, Barriers);

  if (Barriers.empty()) {
    HIPSYCL_DEBUG_INFO << "No splitter found." << std::endl;
    return Changed;
  }

  std::size_t BC = 0;
  F->print(llvm::outs());
  for (auto *Barrier : Barriers) {
    Changed = true;
    ++BC;

    HIPSYCL_DEBUG_INFO << "Found splitter at " << Barrier->getCalledFunction()->getName() << std::endl;

    HIPSYCL_DEBUG_INFO << "Found header: " << L->getHeader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found pre-header: " << L->getLoopPreheader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found exit block: " << L->getExitBlock() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found latch block: " << L->getLoopLatch() << std::endl;

    auto *OldBlock = Barrier->getParent();
    if (LI.getLoopFor(OldBlock) != L) {
      HIPSYCL_DEBUG_ERROR << "Barrier must be directly in item loop for now." << std::endl;
      continue;
    }

    llvm::Loop *ParentLoop = L->getParentLoop();
    llvm::BasicBlock *Header = L->getHeader();
    llvm::BasicBlock *PreHeader = L->getLoopPreheader();
    llvm::BasicBlock *ExitBlock = L->getExitBlock();
    llvm::BasicBlock *Latch = L->getLoopLatch();

    if (Latch == OldBlock) {
      Latch = makeLatch(L, OldBlock, LI, DT);
    }

    if (isInConditional(Barrier, DT, Latch)) {
      HIPSYCL_DEBUG_INFO << "is in conditional" << std::endl;
    }

    llvm::SmallPtrSet<llvm::Instruction *, 8> DependingInsts;
    llvm::SmallVector<llvm::BasicBlock *, 2> BaseBlocks = {Header};
    llvm::SmallVector<llvm::BasicBlock *, 8> SearchBlocks;
    //    DT.getDescendants(header, searchBlocks);
    fillDescendantsExcl(Header, Latch, DT, SearchBlocks);
    //    for (auto *BB : searchBlocks) {
    //      BB->print(llvm::errs());
    //    }
    if (fillDependingInsts(BaseBlocks, SearchBlocks, DependingInsts)) {
      HIPSYCL_DEBUG_INFO << "has dependencies from header" << std::endl;

      for (auto *I : DependingInsts) {
        I->print(llvm::outs());
        llvm::outs() << "\n";
      }
    } else {
      HIPSYCL_DEBUG_WARNING << "what's wrong with you.. empty work item?!" << std::endl;
    }

    llvm::Loop &NewLoop = *LI.AllocateLoop();
    ParentLoop->addChildLoop(&NewLoop);

    llvm::ValueToValueMapTy VMap;
    VMap[PreHeader] = Header;

    const std::string BlockNameSuffix = "split" + std::to_string(BC);
    llvm::ClonedCodeInfo ClonedCodeInfo;
    auto *NewHeader = llvm::CloneBasicBlock(Header, VMap, BlockNameSuffix, F, &ClonedCodeInfo, nullptr);
    VMap[Header] = NewHeader;
    NewLoop.addBlockEntry(NewHeader);
    NewLoop.moveToHeader(NewHeader);

    auto *NewLatch = llvm::CloneBasicBlock(Latch, VMap, BlockNameSuffix, F, &ClonedCodeInfo, nullptr);
    VMap[Latch] = NewLatch;
    NewLoop.addBlockEntry(NewLatch);

    auto *NewBlock = llvm::SplitBlock(OldBlock, Barrier, &DT, &LI, nullptr, OldBlock->getName() + BlockNameSuffix);
    L->removeBlockFromLoop(NewBlock);
    NewLoop.addBlockEntry(NewBlock);
    Barrier->eraseFromParent();
    DependingInsts.erase(Barrier);

    // connect new loop
    NewHeader->getTerminator()->setSuccessor(0, NewBlock);
    NewHeader->getTerminator()->setSuccessor(1, ExitBlock);
    DT.addNewBlock(NewHeader, Header);
    DT.changeImmediateDominator(NewBlock, NewHeader);
    DT.changeImmediateDominator(ExitBlock, NewHeader);

    llvm::SmallVector<llvm::BasicBlock *, 2> Preds{llvm::pred_begin(Latch), llvm::pred_end(Latch)};
    llvm::BasicBlock *Ncd = nullptr;
    for (auto *Pred : Preds) {
      std::size_t SuccIdx = 0;
      for (auto *Succ : llvm::successors(Pred)) {
        if (Succ == Latch)
          break;
        ++SuccIdx;
      }
      Ncd = Ncd ? DT.findNearestCommonDominator(Ncd, Pred) : Pred;

      Pred->getTerminator()->setSuccessor(SuccIdx, NewLatch);
    }
    DT.addNewBlock(NewLatch, Ncd);

    NewLatch->getTerminator()->setSuccessor(0, NewHeader);
    //    DT.changeImmediateDominator(newHeader, newLatch);

    // fix old loop
    Header->getTerminator()->setSuccessor(1, NewHeader);
    OldBlock->getTerminator()->setSuccessor(0, Latch);
    DT.changeImmediateDominator(Latch, OldBlock);
    //    DT.changeImmediateDominator(header, newHeader);

    for (auto *SubLoop : L->getSubLoops()) {
      auto *NewParent = LI.getLoopFor(SubLoop->getLoopPreheader());
      if (NewParent != L) {
        HIPSYCL_DEBUG_INFO << "new parent for subloop: " << NewParent << std::endl;
        L->removeChildLoop(SubLoop);
        NewParent->addChildLoop(SubLoop);
      }
    }

    llvm::SmallVector<llvm::BasicBlock *, 8> BbToRemap;
    DT.getDescendants(NewHeader, BbToRemap);
    HIPSYCL_DEBUG_INFO << "BLOCKS TO REMAP " << BbToRemap.size();
    llvm::SmallVector<llvm::BasicBlock *, 8> SBs;
    fillDescendantsExcl(Header, NewHeader, DT, SBs);
    SBs.erase(std::find(SBs.begin(), SBs.end(), Latch));

    llvm::SmallPtrSet<llvm::Instruction *, 8> InstsToCopy;
    llvm::SmallPtrSet<llvm::Instruction *, 8> InstsInNew;
    for (auto *I : DependingInsts) {
      if (std::find(BbToRemap.begin(), BbToRemap.end(), I->getParent()) != BbToRemap.end())
        InstsInNew.insert(I);
    }
    fillDependingInsts(InstsInNew, SBs, InstsToCopy);
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "insts to copy: "
                 << "\n";
    llvm::SmallVector<llvm::Instruction *, 8> InstsToCopySorted{InstsToCopy.begin(), InstsToCopy.end()};
    std::sort(InstsToCopySorted.begin(), InstsToCopySorted.end(),
              [&](auto LHS, auto RHS) { return DT.dominates(LHS, RHS); });
    //    MemDepP.releaseMemory();
    //    MemDepP.getMemDep();
    llvm::Instruction *InsPt = &*NewBlock->getFirstInsertionPt();
    for (auto *I : InstsToCopySorted) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO;
      I->print(llvm::outs());
      llvm::outs() << "\n";
      auto *ClonedI = I->clone();
      VMap[I] = ClonedI;
      //      if(insPt)
      //        clonedI->insertAfter(insPt);
      //      else
      ClonedI->insertBefore(InsPt);
      //      insPt = clonedI;
    }

    llvm::remapInstructionsInBlocks(BbToRemap, VMap);

    for (auto *Block : L->getParentLoop()->blocks()) {
      if (!Block->getParent())
        Block->print(llvm::errs());
    }
    HIPSYCL_DEBUG_INFO << "new loopx.. " << &NewLoop << " with parent " << NewLoop.getParentLoop() << std::endl;
    DT.print(llvm::errs());
    L = updateDtAndLi(LI, DT, NewLatch, *L->getHeader()->getParent());
    DT.print(llvm::errs());
    LoopAdder(*L);

    HIPSYCL_DEBUG_INFO << "new loop.. " << L << " with parent " << L->getParentLoop() << std::endl;

    //    HIPSYCL_DEBUG_INFO << "new exit block: " << LIW.getLoopInfo().getLoopFor(newBlock)->getExitBlock() <<
    //    std::endl; HIPSYCL_DEBUG_INFO << "old exit block: " << L->getExitBlock() << std::endl;

    for (auto *Block : L->getParentLoop()->blocks()) {
      llvm::SimplifyInstructionsInBlock(Block);
    }

    llvm::simplifyLoop(L->getParentLoop(), &DT, &LI, &SE, &AC, nullptr, false);
    F->viewCFG();
    //    DTW.getDomTree().viewGraph();

    if (llvm::verifyFunction(*F, &llvm::errs())) {
      HIPSYCL_DEBUG_ERROR << "function verification failed" << std::endl;
    }
  }
  F->print(llvm::outs());
  return Changed;
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
