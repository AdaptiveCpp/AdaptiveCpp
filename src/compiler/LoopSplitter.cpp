#include "hipSYCL/compiler/LoopSplitter.hpp"

#include "hipSYCL/common/debug.hpp"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &ost, const llvm::StringRef &strRef) {
  return ost << strRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  // cached, should not need to rerun..
  if (splitterFuncs_)
    return false;
  splitterFuncs_ = llvm::SmallPtrSet<llvm::Function *, 2>{};

  for (auto &I : F.getParent()->globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get());
        auto *FUNC = llvm::dyn_cast<llvm::Function>(CS->getOperand(0)->getOperand(0));
        auto *AnnotationGL = llvm::dyn_cast<llvm::GlobalVariable>(CS->getOperand(1)->getOperand(0));
        llvm::StringRef annotation =
            llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())->getAsCString();
        if (annotation.compare(SplitterAnnotation) == 0) {
          splitterFuncs_->insert(FUNC);
          HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << FUNC->getName() << "\n";
        }
      }
    }
  }
  return false;
}

namespace {

llvm::Loop *UpdateDTAndLI(llvm::LoopInfoWrapperPass &LIW, llvm::DominatorTreeWrapperPass &DTW,
                          const llvm::BasicBlock *B, llvm::Function &F) {
  llvm::Loop *L;
  DTW.releaseMemory();
  DTW.getDomTree().recalculate(F);
  LIW.releaseMemory();
  LIW.getLoopInfo().analyze(DTW.getDomTree());
  L = LIW.getLoopInfo().getLoopFor(B);
  return L;
}

bool InlineCallTree(llvm::CallBase *CI, const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA) {
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
                             const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA) {
  bool changed = false;
  bool lastChanged = false;

  do {
    lastChanged = false;
    for (auto &I : BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() && splitterCallers.find(callI->getCalledFunction()) != splitterCallers.end()) {
          lastChanged = InlineCallTree(callI, SAA);
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
                       const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA, llvm::LoopInfoWrapperPass &LIW,
                       llvm::DominatorTreeWrapperPass &DTW) {
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
      L = UpdateDTAndLI(LIW, DTW, B, F);
    }
  } while (lastChanged);

  return changed;
}

//! \pre \a F is not recursive!
// todo: have a recursive-ness termination
bool FillTransitiveSplitterCallers(llvm::Function &F, const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter) {
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

bool FillTransitiveSplitterCallers(llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA,
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

void FindAllSplitterCalls(const llvm::Loop &L, const hipsycl::compiler::SplitterAnnotationAnalysisLegacy &SAA,
                          llvm::SmallVector<llvm::CallBase *, 8> &barriers) {
  for (auto *BB : L.getBlocks()) {
    for (auto &I : *BB) {
      if (auto *callI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (callI->getCalledFunction() && SAA.IsSplitterFunc(callI->getCalledFunction()))
          barriers.push_back(callI);
      }
    }
  }
}
} // namespace

bool hipsycl::compiler::LoopSplitAtBarrierPassLegacy::runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) {
  if (!(*L->block_begin())->getParent()->getName().startswith(".omp_outlined")) {
    // are we in kernel?
    return false;
  }

  if (L->getLoopDepth() != 2) {
    // only second-level loop have to be considered as work-item loops -> must be using collapse on multi-dim kernels
    HIPSYCL_DEBUG_INFO << "Not work-item loop!" << std::endl;
    return false;
  }
  auto &LIW = getAnalysis<llvm::LoopInfoWrapperPass>();
  auto &DTW = getAnalysis<llvm::DominatorTreeWrapperPass>();
  auto &SE = getAnalysis<llvm::ScalarEvolutionWrapperPass>().getSE();
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>();

  llvm::Function *F = L->getBlocks()[0]->getParent();

  llvm::AssumptionCache AC(*F);

  llvm::SmallPtrSet<llvm::Function *, 8> splitterCallers;
  if (!FillTransitiveSplitterCallers(*L, SAA, splitterCallers)) {
    HIPSYCL_DEBUG_INFO << "Transitively no splitter found." << std::endl;
    return false;
  }

  bool changed = InlineCallsInLoop(L, splitterCallers, SAA, LIW, DTW);

  llvm::SmallVector<llvm::CallBase *, 8> barriers;
  FindAllSplitterCalls(*L, SAA, barriers);

  if (barriers.empty()) {
    HIPSYCL_DEBUG_INFO << "No splitter found." << std::endl;
    return changed;
  }

  std::size_t bC = 0;
  llvm::LoopInfo &LI = LIW.getLoopInfo();
  llvm::DominatorTree &DT = DTW.getDomTree();
  for (auto *barrier : barriers) {
    changed = true;
    HIPSYCL_DEBUG_INFO << "Found splitter at " << barrier->getCalledFunction()->getName() << std::endl;

    HIPSYCL_DEBUG_INFO << "Found header: " << L->getHeader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found pre-header: " << L->getLoopPreheader() << std::endl;
    HIPSYCL_DEBUG_INFO << "Found exit block: " << L->getExitBlock() << std::endl;

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

    llvm::Loop &newLoop = *LI.AllocateLoop();
    parentLoop->addChildLoop(&newLoop);
    LPM.addLoop(newLoop);

    llvm::ValueToValueMapTy vMap;
    vMap[preHeader] = header;

    llvm::ClonedCodeInfo clonedCodeInfo;
    auto *newHeader =
        llvm::CloneBasicBlock(header, vMap, llvm::Twine("split") + llvm::Twine(bC), F, &clonedCodeInfo, nullptr);
    vMap[header] = newHeader;
    newLoop.addBlockEntry(newHeader);
    newLoop.moveToHeader(newHeader);

    auto *newLatch =
        llvm::CloneBasicBlock(latch, vMap, llvm::Twine("split") + llvm::Twine(bC), F, &clonedCodeInfo, nullptr);
    vMap[latch] = newLatch;
    newLoop.addBlockEntry(newLatch);

    auto *newBlock = llvm::SplitBlock(oldBlock, barrier, &DT, &LI);
    L->removeBlockFromLoop(newBlock);
    newLoop.addBlockEntry(newBlock);
    barrier->eraseFromParent();

    // connect new loop
    newHeader->getTerminator()->setSuccessor(0, newBlock);
    newHeader->getTerminator()->setSuccessor(1, exitBlock);

    llvm::SmallVector<llvm::BasicBlock *, 2> preds{llvm::pred_begin(latch), llvm::pred_end(latch)};
    for (auto *pred : preds) {
      std::size_t succIdx = 0;
      for (auto *succ : llvm::successors(pred)) {
        if (succ == latch)
          break;
        ++succIdx;
      }
      pred->getTerminator()->setSuccessor(succIdx, newLatch);
    }

    newLatch->getTerminator()->setSuccessor(0, newHeader);

    // fix old loop
    header->getTerminator()->setSuccessor(1, newHeader);
    oldBlock->getTerminator()->setSuccessor(0, latch);

    for (auto *subLoop : L->getSubLoops()) {
      auto *newParent = LI.getLoopFor(subLoop->getLoopPreheader());
      if (newParent != L) {
        HIPSYCL_DEBUG_INFO << "new parent for subloop: " << newParent << std::endl;
        L->removeChildLoop(subLoop);
        newParent->addChildLoop(subLoop);
      }
    }

    llvm::SmallVector<llvm::BasicBlock *, 8> bbToRemap{newLoop.block_begin(), newLoop.block_end()};
    llvm::remapInstructionsInBlocks(bbToRemap, vMap);

    for (auto *block : L->getParentLoop()->blocks()) {
      if (!block->getParent())
        block->print(llvm::errs());
    }

    L = UpdateDTAndLI(LIW, DTW, L->getHeader(), *L->getHeader()->getParent());

    HIPSYCL_DEBUG_INFO << "new exit block: " << LIW.getLoopInfo().getLoopFor(newBlock)->getExitBlock() << std::endl;
    HIPSYCL_DEBUG_INFO << "old exit block: " << L->getExitBlock() << std::endl;

    llvm::simplifyLoop(L->getParentLoop(), &DTW.getDomTree(), &LIW.getLoopInfo(), &SE, &AC, nullptr, false);
  }
  return changed;
}

void hipsycl::compiler::LoopSplitAtBarrierPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

char hipsycl::compiler::SplitterAnnotationAnalysisLegacy::ID = 0;
char hipsycl::compiler::LoopSplitAtBarrierPassLegacy::ID = 0;
