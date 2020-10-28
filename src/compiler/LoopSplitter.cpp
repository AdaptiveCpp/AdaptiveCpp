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
  llvm::Loop *IL = L;

  auto &LIW = getAnalysis<llvm::LoopInfoWrapperPass>();
  auto &DTW = getAnalysis<llvm::DominatorTreeWrapperPass>();
  //  auto &SE = getAnalysis<llvm::ScalarEvolutionWrapperPass>().getSE();
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>();

  llvm::SmallPtrSet<llvm::Function *, 8> splitterCallers;
  if (!FillTransitiveSplitterCallers(*IL, SAA, splitterCallers)) {
    HIPSYCL_DEBUG_INFO << "Transitively no splitter found." << std::endl;
    return false;
  }

  bool changed = InlineCallsInLoop(IL, splitterCallers, SAA, LIW, DTW);

  llvm::SmallVector<llvm::CallBase *, 8> barriers;
  FindAllSplitterCalls(*IL, SAA, barriers);

  if (barriers.empty()) {
    HIPSYCL_DEBUG_INFO << "No splitter found." << std::endl;
    return changed;
  }
  barriers[0]->getParent()->getParent()->print(llvm::errs());
  barriers[0]->getParent()->getParent()->viewCFG();

  for (auto *barrier : barriers) {
    changed = true;
    HIPSYCL_DEBUG_INFO << "Found barrier at " << barrier->getCalledFunction()->getName() << std::endl;

    // todo: split loop at instruction.
  }

  return changed;
}

void hipsycl::compiler::LoopSplitAtBarrierPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  llvm::getLoopAnalysisUsage(AU);

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

char hipsycl::compiler::SplitterAnnotationAnalysisLegacy::ID = 0;
char hipsycl::compiler::LoopSplitAtBarrierPassLegacy::ID = 0;
