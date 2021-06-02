#include "hipSYCL/compiler/LoopSplitter.hpp"

#include "hipSYCL/common/debug.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
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
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

namespace hipsycl {
namespace compiler {
static constexpr size_t NumArrayElements = 1024;
struct MDKind {
  static constexpr const char Arrayified[] = "hipSYCL.arrayified";
  static constexpr const char InnerLoop[] = "hipSYCL.loop.inner";
  static constexpr const char WorkItemLoop[] = "hipSYCL.loop.workitem";
};
} // namespace compiler
} // namespace hipsycl

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &Ost, const llvm::StringRef &StrRef) {
  return Ost << StrRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationInfo::analyzeModule(const llvm::Module &Module) {
  for (auto &I : Module.globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto *OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get());
        auto *F = llvm::dyn_cast<llvm::Function>(CS->getOperand(0)->getOperand(0));
        auto *AnnotationGL = llvm::dyn_cast<llvm::GlobalVariable>(CS->getOperand(1)->getOperand(0));
        llvm::StringRef Annotation =
            llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())->getAsCString();
        if (Annotation.compare(SplitterAnnotation) == 0) {
          SplitterFuncs.insert(F);
          HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << F->getName() << "\n";
        } else if (Annotation.compare(KernelAnnotation) == 0) {
          NDKernels.insert(F);
          HIPSYCL_DEBUG_INFO << "Found kernel annotated function " << F->getName() << "\n";
        }
      }
    }
  }
  return false;
}

hipsycl::compiler::SplitterAnnotationInfo::SplitterAnnotationInfo(const llvm::Module &Module) { analyzeModule(Module); }

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  if (SplitterAnnotation_)
    return false;
  SplitterAnnotation_ = SplitterAnnotationInfo{*F.getParent()};
  return false;
}

hipsycl::compiler::SplitterAnnotationAnalysis::Result
hipsycl::compiler::SplitterAnnotationAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
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

bool inlineSplitterCallTree(llvm::CallBase *CI) {
  if (CI->getCalledFunction()->isIntrinsic())
    return false;

  // needed to be valid for success log
  const auto CalleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR <= 10
  llvm::InlineResult ILR = llvm::InlineFunction(CI, IFI, nullptr);
  if (!static_cast<bool>(ILR)) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.message
                 << "'\n";
#else
  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI, nullptr);
  if (!ILR.isSuccess()) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << CalleeName << ">: '"
                 << ILR.getFailureReason() << "'\n";
#endif
    return false;
  }

  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "LoopSplitter inlined function <"
                                          << CalleeName << ">\n";)
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
        if (CallI->getCalledFunction() && SplitterCallers.find(CallI->getCalledFunction()) != SplitterCallers.end() &&
            !SAA.isSplitterFunc(CallI->getCalledFunction())) {
          LastChanged = inlineSplitterCallTree(CallI);
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

bool inlineCallsInBasicBlock(llvm::BasicBlock &BB) {
  bool Changed = false;
  bool LastChanged = false;

  do {
    LastChanged = false;
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction()) {
          LastChanged = inlineSplitterCallTree(CallI);
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
bool inlineCallsInLoop(llvm::Loop *&L, llvm::LoopInfo &LI, llvm::DominatorTree &DT) {
  bool Changed = false;
  bool LastChanged = false;

  llvm::BasicBlock *B = L->getBlocks()[0];
  llvm::Function &F = *B->getParent();

  do {
    LastChanged = false;
    for (auto *BB : L->getBlocks()) {
      LastChanged = inlineCallsInBasicBlock(*BB);
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
bool isInConditional(const llvm::CallBase *BarrierI, const llvm::DominatorTree &DT, const llvm::BasicBlock *Latch) {
  return !DT.properlyDominates(BarrierI->getParent(), Latch);
}

bool fillDescendantsExcl(const llvm::BasicBlock *Root, const llvm::ArrayRef<const llvm::BasicBlock *> Excl,
                         const llvm::DominatorTree &DT, llvm::SmallVectorImpl<llvm::BasicBlock *> &SearchBlocks) {
  const auto *RootNode = DT.getNode(Root);
  if (!RootNode)
    return false;

  llvm::SmallVector<const llvm::DomTreeNodeBase<llvm::BasicBlock> *, 8> WL;
  WL.append(RootNode->begin(), RootNode->end());

  while (!WL.empty()) {
    const llvm::DomTreeNodeBase<llvm::BasicBlock> *N = WL.pop_back_val();
    if (std::find(Excl.begin(), Excl.end(), N->getBlock()) == Excl.end()) {
      WL.append(N->begin(), N->end());
      SearchBlocks.push_back(N->getBlock());
    }
  }
  return !SearchBlocks.empty();
}

bool fillBlocksInBranch(const llvm::DomTreeNodeBase<llvm::BasicBlock> *First, const llvm::BasicBlock *Merge,
                        llvm::SmallVectorImpl<llvm::BasicBlock *> &Blocks) {
  llvm::SmallVector<const llvm::DomTreeNodeBase<llvm::BasicBlock> *, 8> WL;
  WL.push_back(First);
  while (!WL.empty()) {
    const llvm::DomTreeNodeBase<llvm::BasicBlock> *N = WL.pop_back_val();
    if (N->getBlock() != Merge) {
      HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << N->getBlock()->getName() << "\n";)
      Blocks.push_back(N->getBlock());
      WL.append(N->begin(), N->end());
    }
  }
  return !Blocks.empty();
}

bool findBlocksInBranch(const llvm::BasicBlock *Cond, const llvm::BasicBlock *Merge, const llvm::DominatorTree &DT,
                        llvm::SmallVectorImpl<llvm::BasicBlock *> &Blocks1,
                        llvm::SmallVectorImpl<llvm::BasicBlock *> &Blocks2) {
  if (Cond->getTerminator()->getNumSuccessors() != 2)
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_ERROR << "Must be dual child branch\n";

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "cond blocks " << Cond->getName() << "\n";
  fillBlocksInBranch(DT.getNode(Cond->getTerminator()->getSuccessor(0)), Merge, Blocks1);
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "cond blocks2 " << Cond->getName() << "\n";
  fillBlocksInBranch(DT.getNode(Cond->getTerminator()->getSuccessor(1)), Merge, Blocks2);
  return !Blocks1.empty() || !Blocks2.empty();
}

struct Condition {
  const llvm::BasicBlock *Cond;
  const llvm::BasicBlock *Merge = nullptr;
  llvm::SmallVector<llvm::BasicBlock *, 4> BlocksLeft;
  llvm::SmallVector<llvm::BasicBlock *, 4> BlocksRight;
  const llvm::BasicBlock *InnerCondLeft = nullptr;
  const llvm::BasicBlock *InnerCondRight = nullptr;
  const llvm::BasicBlock *ParentCond = nullptr;

  Condition() : Cond(nullptr) {
    assert(false && "should never be called"); // but is required for the operator[] of DenseMap
  }
  Condition(const llvm::BasicBlock *CondP) : Cond(CondP) {}

  ~Condition() {
    Cond = nullptr;
    Merge = nullptr;
  }
};

void findIfConditionInner(
    const llvm::BasicBlock *Root, const llvm::ArrayRef<const llvm::BasicBlock *> Terminals,
    const llvm::SmallVectorImpl<llvm::Loop *> &Loops, const llvm::DominatorTree &DT,
    llvm::SmallDenseMap<const llvm::BasicBlock *, int, 8> &LookedAtBBs, llvm::SmallVectorImpl<Condition *> &BranchStack,
    llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> &BranchCondAndMerge) {
  llvm::BasicBlock *IfThen = nullptr, *IfElse = nullptr;

  const auto NumSuccessors = Root->getTerminator()->getNumSuccessors();
  if (NumSuccessors > 1) {
    const auto *LoopIt =
        std::find_if(Loops.begin(), Loops.end(), [Root](auto *Loop) { return Loop->getHeader() == Root; });
    auto Pair = BranchCondAndMerge.try_emplace(Root, std::make_unique<Condition>(Root));

    if (LoopIt == Loops.end()) {
      BranchStack.push_back(Pair.first->second.get());
    } else {
      Pair.first->second->Merge = Root;
      Pair.first->second->BlocksLeft.append((*LoopIt)->block_begin() + 1, (*LoopIt)->block_end());
    }
  }

  for (size_t S = 0; S < NumSuccessors; ++S) {
    auto *Successor = Root->getTerminator()->getSuccessor(S);
    if (std::find(Terminals.begin(), Terminals.end(), Successor) == Terminals.end()) {
      auto &Visitations = LookedAtBBs[Successor];
      Visitations++;
      if (Successor->hasNPredecessorsOrMore(Visitations + 1) &&
          std::none_of(Loops.begin(), Loops.end(),
                       [Successor](auto *Loop) { return Loop->getHeader() == Successor; })) {
        auto *Branch = BranchStack.pop_back_val();
        if (Branch->Cond != Successor) {
          Branch->Merge = Successor;

          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << Branch->Cond->getName() << ">>" << Successor->getName() << "\n";
          findBlocksInBranch(Branch->Cond, Successor, DT, Branch->BlocksLeft, Branch->BlocksRight);
          llvm::outs().flush();
        }
      }
      if (Visitations == 1) {
        findIfConditionInner(Successor, Terminals, Loops, DT, LookedAtBBs, BranchStack, BranchCondAndMerge);
      }
    }
  }
}

llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8>
findIfCondition(const llvm::BasicBlock *Root, const llvm::ArrayRef<const llvm::BasicBlock *> Terminals,
                const llvm::SmallVectorImpl<llvm::Loop *> &Loops, const llvm::DominatorTree &DT) {
  llvm::SmallDenseMap<const llvm::BasicBlock *, int, 8> LookedAtBBs;
  llvm::SmallVector<Condition *, 8> BranchStack;
  llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> BranchCondAndMerge;

  for (auto *L : Loops) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "L Header " << L->getHeader()->getName() << "\n";
  }

  LookedAtBBs[Root] = 0;
  const auto NumSuccessors = Root->getTerminator()->getNumSuccessors();
  for (size_t S = 0; S < NumSuccessors; ++S) {
    auto *Successor = Root->getTerminator()->getSuccessor(S);
    if (std::find(Terminals.begin(), Terminals.end(), Successor) == Terminals.end()) {
      findIfConditionInner(Successor, Terminals, Loops, DT, LookedAtBBs, BranchStack, BranchCondAndMerge);
    }
  }
  for (auto &CondPair : BranchCondAndMerge) {
    Condition *Cond = CondPair.second.get();
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "cond " << Cond->Cond->getName() << "\n";
    auto *RightIt = std::find_if(Cond->BlocksRight.begin(), Cond->BlocksRight.end(), [&BranchCondAndMerge](auto *BB) {
      return BranchCondAndMerge.find(BB) != BranchCondAndMerge.end();
    });
    auto *LeftIt = std::find_if(Cond->BlocksLeft.begin(), Cond->BlocksLeft.end(), [&BranchCondAndMerge](auto *BB) {
      return BranchCondAndMerge.find(BB) != BranchCondAndMerge.end();
    });
    if (RightIt != Cond->BlocksRight.end()) {
      BranchCondAndMerge[*RightIt]->ParentCond = Cond->Cond;
      Cond->InnerCondRight = *RightIt;
    }
    if (LeftIt != Cond->BlocksLeft.end()) {
      BranchCondAndMerge[*LeftIt]->ParentCond = Cond->Cond;
      Cond->InnerCondLeft = *LeftIt;
    }
  }
  for (auto &CondPair : BranchCondAndMerge) {
    auto *Branch = CondPair.second.get();
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "cond " << Branch->Cond->getName()
                 << " parent: " << (Branch->ParentCond ? Branch->ParentCond->getName() : "")
                 << " left child cond: " << (Branch->InnerCondLeft ? Branch->InnerCondLeft->getName() : "")
                 << " right child cond " << (Branch->InnerCondRight ? Branch->InnerCondRight->getName() : "") << "\n";
  }
  return BranchCondAndMerge;
}

bool findBaseBlocks(
    const llvm::BasicBlock *Root, const llvm::BasicBlock *Latch, const llvm::ArrayRef<const llvm::BasicBlock *> Excl,
    const llvm::DominatorTree &DT, llvm::SmallVectorImpl<llvm::BasicBlock *> &BaseBlocks,
    const llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> &CondsAndMerges) {
  const auto *RootNode = DT.getNode(Root);
  if (!RootNode)
    return false;

  llvm::SmallVector<const llvm::DomTreeNodeBase<llvm::BasicBlock> *, 8> WL;
  WL.append(RootNode->begin(), RootNode->end());

  while (!WL.empty()) {
    const llvm::DomTreeNodeBase<llvm::BasicBlock> *N = WL.pop_back_val();
    if (std::find(Excl.begin(), Excl.end(), N->getBlock()) != Excl.end())
      continue;
    if (auto CondIt = CondsAndMerges.find(const_cast<const llvm::BasicBlock *>(N->getBlock()));
        CondIt != CondsAndMerges.end()) {
      const auto &CondBlocks = CondIt->second->BlocksLeft;
      const auto &CondBlocks2 = CondIt->second->BlocksRight;
      if (auto *ExclIt =
              std::find_if(CondBlocks.begin(), CondBlocks.end(),
                           [&Excl](auto *BB) { return std::find(Excl.begin(), Excl.end(), BB) != Excl.end(); });
          ExclIt != CondBlocks.end()) {
        if (std::any_of(CondBlocks2.begin(), CondBlocks2.end(),
                        [&Excl](auto *BB) { return std::find(Excl.begin(), Excl.end(), BB) != Excl.end(); }))
          llvm::outs() << HIPSYCL_DEBUG_PREFIX_ERROR << "The other branch must not also contain an end\n";
        WL.push_back(DT.getNode(CondIt->second->Cond->getTerminator()->getSuccessor(0)));
      } else if (auto *ExclIt2 =
                     std::find_if(CondBlocks2.begin(), CondBlocks2.end(),
                                  [&Excl](auto *BB) { return std::find(Excl.begin(), Excl.end(), BB) != Excl.end(); });
                 ExclIt2 != CondBlocks2.end()) {
        if (std::any_of(CondBlocks.begin(), CondBlocks.end(),
                        [&Excl](auto *BB) { return std::find(Excl.begin(), Excl.end(), BB) != Excl.end(); }))
          llvm::outs() << HIPSYCL_DEBUG_PREFIX_ERROR << "The other branch must not also contain an end2\n";
        WL.push_back(DT.getNode(CondIt->second->Cond->getTerminator()->getSuccessor(1)));
      } else {
        for (auto *CN : N->children()) {
          if (std::find(Excl.begin(), Excl.end(), CN->getBlock()) == Excl.end())
            WL.push_back(CN);
        }
      }
      BaseBlocks.push_back(N->getBlock());
    } else {
      WL.append(N->begin(), N->end());
      BaseBlocks.push_back(N->getBlock());
    }
  }

  return !BaseBlocks.empty();
}

void findDependenciesBetweenBlocks(const llvm::SmallVectorImpl<llvm::BasicBlock *> &BaseBlocks,
                                   const llvm::SmallVectorImpl<llvm::BasicBlock *> &DependingBlocks,
                                   llvm::SmallPtrSetImpl<llvm::Instruction *> &DependingInsts,
                                   llvm::SmallPtrSetImpl<llvm::Instruction *> &DependedUponValues) {
  llvm::SmallPtrSet<llvm::Instruction *, 8> WL;
  for (auto *B : BaseBlocks)
    for (auto &I : *B)
      WL.insert(&I);
  for (auto *V : WL) {
    for (auto *U : V->users()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (std::find(DependingBlocks.begin(), DependingBlocks.end(), I->getParent()) != DependingBlocks.end()) {
          if (auto *BCI = llvm::dyn_cast<llvm::BitCastInst>(V)) {
            if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(BCI->getOperand(0))) {
              auto *BCICloned = BCI->clone();
              BCICloned->insertBefore(I);
              I->replaceUsesOfWith(BCI, BCICloned);
              V = GEP;
              I = BCICloned;
            }
          }
          DependingInsts.insert(I);
          DependedUponValues.insert(V);
        }
      }
    }
  }
}

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup) {
  if (auto *PresentMD = I->getMetadata(llvm::LLVMContext::MD_access_group)) {
    llvm::SmallVector<llvm::Metadata *, 4> MDs;
    if (PresentMD->getNumOperands() == 0)
      MDs.push_back(PresentMD);
    else
      MDs.append(PresentMD->op_begin(), PresentMD->op_end());
    MDs.push_back(MDAccessGroup);
    auto *CombinedMDAccessGroup = llvm::MDNode::getDistinct(I->getContext(), MDs);
    I->setMetadata(llvm::LLVMContext::MD_access_group, CombinedMDAccessGroup);
  } else
    I->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
}

void promoteAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT, llvm::AssumptionCache &AC) {
  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  while (true) {
    WL.clear();
    for (auto &I : *EntryBlock) {
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (llvm::isAllocaPromotable(Alloca))
          WL.push_back(Alloca);
      }
    }
    if (WL.empty())
      break;
    PromoteMemToReg(WL, DT, &AC);
  }
}

void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::Loop &L, llvm::Value *Idx, const llvm::DominatorTree &DT,
                     llvm::MDNode *MDAccessGroup) {
  auto *MDAlloca =
      llvm::MDNode::get(EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), "hipSYCLLoopState")});

  auto &LoopBlocks = L.getBlocksSet();
  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (llvm::MDNode *MD = Alloca->getMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue; // already arrayificated
      if (!std::all_of(Alloca->user_begin(), Alloca->user_end(), [&LoopBlocks](llvm::User *User) {
            auto *Inst = llvm::dyn_cast<llvm::Instruction>(User);
            return Inst && LoopBlocks.contains(Inst->getParent());
          }))
        continue;
      WL.push_back(Alloca);
    }
  }

  for (auto *I : WL) {
    llvm::IRBuilder AllocaBuilder{I};
    llvm::Type *T = I->getAllocatedType();
    if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
      auto ArrSize = ArrSizeC->getLimitedValue();
      if (ArrSize > 1) {
        T = llvm::ArrayType::get(T, ArrSize);
        llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Caution, alloca was array\n";
      }
    }

    auto *Alloca = AllocaBuilder.CreateAlloca(T, AllocaBuilder.getInt32(hipsycl::compiler::NumArrayElements),
                                              I->getName() + "_alloca");
    Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

    llvm::Instruction *GepIp = nullptr;
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!GepIp || DT.dominates(UI, GepIp))
          GepIp = UI;
      }
    }
    if (GepIp) {
      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEPV = LoadBuilder.CreateInBoundsGEP(Alloca, {Idx}, I->getName() + "_gep");
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(GEPV);
      GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      I->replaceAllUsesWith(GEP);
      I->eraseFromParent();

      for (auto *U : GEP->users()) {
        if (auto *LoadI = llvm::dyn_cast<llvm::LoadInst>(U)) {
          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "llvm.access.group adding to ";
          LoadI->print(llvm::outs());
          llvm::outs() << "\n";
          addAccessGroupMD(LoadI, MDAccessGroup);
        }
      }
    }
  }
}

llvm::AllocaInst *getLoopStateAllocaForLoad(llvm::LoadInst &LInst) {
  llvm::AllocaInst *Alloca = nullptr;
  if (auto *GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(LInst.getPointerOperand())) {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(GEPI->getPointerOperand());
  } else {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(LInst.getPointerOperand());
  }
  if (Alloca && Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
    return Alloca;
  return nullptr;
}

llvm::AllocaInst *arrayifyValue(llvm::Instruction *IPAllocas, llvm::Value *ToArrayify,
                                llvm::Instruction *InsertionPoint, llvm::Value *Idx, llvm::MDNode *MDAccessGroup,
                                llvm::MDTuple *MDAlloca = nullptr) {
  if (!MDAlloca)
    MDAlloca =
        llvm::MDNode::get(IPAllocas->getContext(), {llvm::MDString::get(IPAllocas->getContext(), "hipSYCLLoopState")});

  auto *T = ToArrayify->getType();
  llvm::IRBuilder AllocaBuilder{IPAllocas};
  auto *Alloca = AllocaBuilder.CreateAlloca(T, AllocaBuilder.getInt32(hipsycl::compiler::NumArrayElements),
                                            ToArrayify->getName() + "_alloca");
  Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

  llvm::IRBuilder WriteBuilder{InsertionPoint};
  auto *GEP = WriteBuilder.CreateInBoundsGEP(Alloca, {Idx}, ToArrayify->getName() + "_gep");
  auto *LTStart = WriteBuilder.CreateLifetimeStart(GEP); // todo: calculate size of object.
  LTStart->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
  auto *Store = WriteBuilder.CreateStore(ToArrayify, GEP);
  Store->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
  return Alloca;
}

llvm::LoadInst *loadFromAlloca(llvm::AllocaInst *Alloca, llvm::Value *Idx, llvm::Instruction *InsertBefore,
                               llvm::MDNode *MDAccessGroup, const llvm::Twine &NamePrefix = "") {
  llvm::IRBuilder LoadBuilder{InsertBefore};
  auto *GEP = LoadBuilder.CreateInBoundsGEP(Alloca, {Idx}, NamePrefix + "_lgep");
  auto *Load = LoadBuilder.CreateLoad(GEP, NamePrefix + "_load");
  Load->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);

  return Load;
}

// If InsertBefore = nullptr, ToStore must be an llvm::Instruction.
// The insertion point will be immediately after ToStore then.
void storeToAlloca(llvm::Value &ToStore, llvm::AllocaInst &DstAlloca, llvm::Value &Idx, llvm::MDNode *MDAccessGroup,
                   llvm::Instruction *InsertBefore = nullptr) {
  if (!InsertBefore) {
    auto *ToStoreI = llvm::cast<llvm::Instruction>(&ToStore); // must be inst, as InsertBefore null
    InsertBefore = &*(++ToStoreI->getIterator());
  }
  assert(InsertBefore && "must have insertion point");

  llvm::IRBuilder WriteBuilder{InsertBefore};
  auto *GEP = WriteBuilder.CreateInBoundsGEP(&DstAlloca, {&Idx}, ToStore.getName() + "_gep");
  auto *Store = WriteBuilder.CreateStore(&ToStore, GEP);
  Store->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
}

llvm::AllocaInst *arrayifyInstruction(llvm::Instruction *IPAllocas, llvm::Instruction *ToArrayify, llvm::Value *Idx,
                                      llvm::MDNode *MDAccessGroup, llvm::MDTuple *MDAlloca = nullptr) {
  llvm::Instruction *InsertionPoint = &*(++ToArrayify->getIterator());

  return arrayifyValue(IPAllocas, ToArrayify, InsertionPoint, Idx, MDAccessGroup, MDAlloca);
}

void arrayifyDependedUponValues(llvm::Instruction *IPAllocas, llvm::Value *Idx,
                                const llvm::SmallPtrSet<llvm::Instruction *, 8> &DependedUponValues,
                                llvm::MDNode *MDAccessGroup,
                                llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap) {
  auto *MDAlloca =
      llvm::MDNode::get(IPAllocas->getContext(), {llvm::MDString::get(IPAllocas->getContext(), "hipSYCLLoopState")});
  for (auto *I : DependedUponValues) {
    if (auto *MD = I->getMetadata(hipsycl::compiler::MDKind::Arrayified))
      continue; // currently just have one MD, so no further value checks

    if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(I)) {
      if (auto *Alloca = getLoopStateAllocaForLoad(*LInst)) {
        ValueAllocaMap[I] = Alloca;
        continue;
      }
    }

    ValueAllocaMap[I] = arrayifyInstruction(IPAllocas, I, Idx, MDAccessGroup, MDAlloca);
  }
}

void copyDgbValues(llvm::Value *From, llvm::Value *To, llvm::Instruction *InsertBefore) {
  llvm::SmallVector<llvm::DbgValueInst *, 1> DbgValues;
  llvm::findDbgValues(DbgValues, From);
  if (!DbgValues.empty()) {
    auto *DbgValue = DbgValues.back();
    llvm::DIBuilder DbgBuilder{*InsertBefore->getParent()->getParent()->getParent()};
    DbgBuilder.insertDbgValueIntrinsic(To, DbgValue->getVariable(), DbgValue->getExpression(), DbgValue->getDebugLoc(),
                                       InsertBefore);
  }
}

void insertDbgValueInIncoming(llvm::PHINode *Phi, llvm::Value *IncomingValue, llvm::BasicBlock *IncomingBlock) {
  llvm::SmallVector<llvm::DbgValueInst *, 2> DbgValues;
  llvm::findDbgValues(DbgValues, Phi);
  if (!DbgValues.empty()) {
    auto *DbgV = DbgValues.back();
    llvm::DIBuilder DbgBuilder{*IncomingBlock->getParent()->getParent()};
    DbgBuilder.insertDbgValueIntrinsic(IncomingValue, DbgV->getVariable(), DbgV->getExpression(), DbgV->getDebugLoc(),
                                       IncomingBlock->getTerminator());
  }
}

void dropDebugLocation(llvm::BasicBlock *BB) {
  for (auto &I : *BB) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
    if (!CI || !llvm::isDbgInfoIntrinsic(CI->getIntrinsicID())) {
#if LLVM_VERSION_MAJOR >= 12
      I.dropLocation();
#else
      I.setDebugLoc({});
#endif
    }
  }
}

llvm::DbgValueInst *getThisPtrDbgValue(llvm::Function *F) {
  for (auto &A : F->args()) {
    llvm::SmallVector<llvm::DbgValueInst *, 2> DbgValues;
    llvm::findDbgValues(DbgValues, &A);
    for (auto *DV : DbgValues) {
      if (DV->getVariable()->getName() == "this") {
        return DV;
      }
    }
  }
  return nullptr;
}

void replaceOperandsWithArrayLoad(llvm::Value *Idx,
                                  const llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap,
                                  const llvm::SmallPtrSet<llvm::Instruction *, 8> &DependingInsts,
                                  llvm::MDNode *MDAccessGroup) {
  for (auto *I : DependingInsts) {
    for (auto &OP : I->operands()) {
      auto *OPV = OP.get();
      if (auto *OPI = llvm::dyn_cast<llvm::GetElementPtrInst>(OPV)) {
        if (OPI->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
          llvm::Instruction *ClonedI = OPI->clone();
          ClonedI->insertBefore(I);
          I->replaceUsesOfWith(OPI, ClonedI); // todo: optimize location and re-usage..
          continue;
        }
      }
      if (auto AllocaIt = ValueAllocaMap.find(OPV); AllocaIt != ValueAllocaMap.end()) {
        auto *Alloca = llvm::cast<llvm::AllocaInst>(AllocaIt->getSecond());
        auto *IP = I;
        // here's probably not the place for this.. as we need this in the pre-header or so of the new loop and not in
        // the old work-item loop.. :(
        if (auto *PhiI = llvm::dyn_cast<llvm::PHINode>(I)) {
          auto *IncomingBB = PhiI->getIncomingBlock(OP);
          IP = IncomingBB->getTerminator();
        }
        llvm::LoadInst *Load = loadFromAlloca(Alloca, Idx, IP, MDAccessGroup, OPV->getName());

        copyDgbValues(OPV, Load, IP);

        I->replaceUsesOfWith(OPV, Load);
      }
    }
  }
}

void replaceOperandsWithArrayLoadInHeader(llvm::Value *Idx,
                                          llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap,
                                          const llvm::SmallPtrSet<llvm::Instruction *, 8> &DependingInsts,
                                          llvm::Instruction *InsertBefore, llvm::ValueToValueMapTy &VMap,
                                          llvm::MDNode *MDAccessGroup) {
  llvm::SmallPtrSet<llvm::Instruction *, 8> AllocasInHeader;
  for (auto *I : DependingInsts) {
    for (auto &OP : I->operands()) {
      auto *OPV = OP.get();
      if (auto *OPI = llvm::dyn_cast<llvm::GetElementPtrInst>(OPV)) {
        if (OPI->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
          llvm::Instruction *ClonedI = OPI->clone();
          ClonedI->insertBefore(I);
          I->replaceUsesOfWith(OPI, ClonedI); // todo: optimize location and re-usage..
          continue;
        }
      }
      if (auto AllocaIt = ValueAllocaMap.find(OPV); AllocaIt != ValueAllocaMap.end()) {
        auto *Alloca = llvm::cast<llvm::AllocaInst>(AllocaIt->getSecond());
        auto *IP = InsertBefore;
        // here's probably not the place for this.. as we need this in the pre-header or so of the new loop and not in
        // the old work-item loop.. :(
        if (auto *PhiI = llvm::dyn_cast<llvm::PHINode>(I)) {
          auto *IncomingBB = PhiI->getIncomingBlock(OP);
          IP = IncomingBB->getTerminator();
          llvm::LoadInst *Load = loadFromAlloca(Alloca, Idx, IP, MDAccessGroup, OPV->getName());
          copyDgbValues(OPV, Load, IP);
          I->replaceUsesOfWith(OPV, Load);
        } else if (!AllocasInHeader.contains(Alloca)) {
          llvm::LoadInst *Load = loadFromAlloca(Alloca, Idx, IP, MDAccessGroup, OPV->getName());

          copyDgbValues(OPV, Load, IP);
          VMap[AllocaIt->first] = Load;
          AllocasInHeader.insert(Alloca);
        }
      }
    }
  }
}

template <class UserType, class Func> bool anyOfUsers(llvm::Value *V, Func &&L) {
  for (auto *U : V->users())
    if (UserType *UT = llvm::dyn_cast<UserType>(U))
      if (L(UT))
        return true;
  return false;
}

template <class UserType, class Func> bool noneOfUsers(llvm::Value *V, Func &&L) {
  return !anyOfUsers<UserType>(V, std::forward<Func>(L));
}

template <class UserType, class Func> bool allOfUsers(llvm::Value *V, Func &&L) {
  return !anyOfUsers<UserType>(V, [L = std::forward<Func>(L)](UserType *UT) { return !L(UT); });
}

void insertLifetimeEndForAllocas(llvm::Function *F, llvm::Value *Idx, llvm::Instruction *IP,
                                 llvm::MDNode *MDAccessGroup) {
  auto &EntryBlock = F->getEntryBlock();
  llvm::IRBuilder Builder{IP};
  auto IsLifetimeEnd = [](auto *CI) {
    return CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_end;
  };

  for (auto &I : EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
        if (noneOfUsers<llvm::GetElementPtrInst>(Alloca, [Alloca, &IsLifetimeEnd](auto *GEP) {
              return anyOfUsers<llvm::CallBase>(GEP, IsLifetimeEnd) ||
                     anyOfUsers<llvm::BitCastInst>(GEP, [Alloca, &IsLifetimeEnd](auto *BCI) {
                       return anyOfUsers<llvm::CallBase>(BCI, IsLifetimeEnd);
                     });
            })) {
          auto *GEP = Builder.CreateInBoundsGEP(Alloca, {Idx}, Alloca->getName() + "_legep");
          auto *LifetimeEnd = Builder.CreateLifetimeEnd(GEP);
          LifetimeEnd->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
        }
      }
    }
  }
}

void arrayifyDependencies(llvm::Function *F, llvm::Value *Idx,
                          llvm::SmallVectorImpl<llvm::BasicBlock *> &ArrfBaseBlocks,
                          llvm::SmallVectorImpl<llvm::BasicBlock *> &ArrfSearchBlocks, llvm::BasicBlock *LoadBlock,
                          llvm::ValueToValueMapTy &VMap, llvm::MDNode *MDAccessGroup) {
  llvm::SmallPtrSet<llvm::Instruction *, 8> ArrfDependingInsts;
  llvm::SmallPtrSet<llvm::Instruction *, 8> ArrfDependedUponValues;
  llvm::DenseMap<llvm::Value *, llvm::Instruction *> ValueAllocaMap;

  HIPSYCL_DEBUG_EXECUTE_INFO(
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "baseblocks:\n"; for (auto *BB
                                                                         : ArrfBaseBlocks) { BB->print(llvm::outs()); }

                                                                    llvm::outs()
                                                                    << HIPSYCL_DEBUG_PREFIX_INFO << "searchblocks:\n";
      for (auto *BB
           : ArrfSearchBlocks) { BB->print(llvm::outs()); })

  findDependenciesBetweenBlocks(ArrfBaseBlocks, ArrfSearchBlocks, ArrfDependingInsts, ArrfDependedUponValues);
  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "depended upon values\n";
                             for (auto *V
                                  : ArrfDependedUponValues) {
                               V->print(llvm::outs());
                               llvm::outs() << "\n";
                             })
  arrayifyDependedUponValues(F->getEntryBlock().getFirstNonPHI(), Idx, ArrfDependedUponValues, MDAccessGroup,
                             ValueAllocaMap);
  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "depending insts\n";
                             for (auto *I
                                  : ArrfDependingInsts) {
                               I->print(llvm::outs());
                               llvm::outs() << "\n";
                             })
  replaceOperandsWithArrayLoadInHeader(Idx, ValueAllocaMap, ArrfDependingInsts, LoadBlock->getFirstNonPHI(), VMap,
                                       MDAccessGroup);
}

bool moveArrayLoadForPhiToIncomingBlock(llvm::BasicBlock *BB) {
  llvm::SmallVector<llvm::PHINode *, 2> Phis;
  for (auto &I : *BB) {
    if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
      Phis.push_back(Phi);
    }
  }
  bool Changed = false;
  for (auto *Phi : Phis) {
    for (auto &OP : Phi->incoming_values()) {
      // todo: check if instructions in header at all
      if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&OP)) {
        GEP->moveBefore(Phi->getIncomingBlock(OP)->getTerminator());
        Changed = true; // todo: remind me, why do we need the GEP here alone again?
      } else if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(&OP)) {
        if (getLoopStateAllocaForLoad(*Load) == nullptr)
          continue; // only do this for loads that load from a loop state alloca.
        Load->moveBefore(Phi->getIncomingBlock(OP)->getTerminator());
        auto *Ptr = Load->getPointerOperand();
        if (auto *GEPL = llvm::dyn_cast<llvm::GetElementPtrInst>(Ptr)) {
          GEPL->moveBefore(Load);
        }
        Changed = true;
      } else if (auto *Const = llvm::dyn_cast<llvm::Constant>(&OP)) {
        insertDbgValueInIncoming(Phi, Const, Phi->getIncomingBlock(OP));
      }
    }
  }
  return Changed;
}

bool isAnnotatedParallel(llvm::Loop *TheLoop) { // from llvm for debugging. Todo: remove again
  llvm::MDNode *DesiredLoopIdMetadata = TheLoop->getLoopID();

  if (!DesiredLoopIdMetadata)
    return false;

  llvm::MDNode *ParallelAccesses = llvm::findOptionMDForLoop(TheLoop, "llvm.loop.parallel_accesses");
  llvm::SmallPtrSet<llvm::MDNode *, 4> ParallelAccessGroups; // For scalable 'contains' check.
  if (ParallelAccesses) {
    for (const llvm::MDOperand &MD : llvm::drop_begin(ParallelAccesses->operands(), 1)) {
      llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(MD.get());
      assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
      ParallelAccessGroups.insert(AccGroup);
    }
  }

  // The loop branch contains the parallel loop metadata. In order to ensure
  // that any parallel-loop-unaware optimization pass hasn't added loop-carried
  // dependencies (thus converted the loop back to a sequential loop), check
  // that all the memory instructions in the loop belong to an access group that
  // is parallel to this loop.
  for (llvm::BasicBlock *BB : TheLoop->blocks()) {
    for (llvm::Instruction &I : *BB) {
      if (!I.mayReadOrWriteMemory())
        continue;

      if (llvm::MDNode *AccessGroup = I.getMetadata(llvm::LLVMContext::MD_access_group)) {
        auto ContainsAccessGroup = [&ParallelAccessGroups](llvm::MDNode *AG) -> bool {
          if (AG->getNumOperands() == 0) {
            assert(llvm::isValidAsAccessGroup(AG) && "Item must be an access group");
            return ParallelAccessGroups.count(AG);
          }

          for (const llvm::MDOperand &AccessListItem : AG->operands()) {
            llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(AccessListItem.get());
            assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
            if (ParallelAccessGroups.count(AccGroup))
              return true;
          }
          return false;
        };

        if (ContainsAccessGroup(AccessGroup))
          continue;
      }
      auto ReturnFalse = [&I]() {
        HIPSYCL_DEBUG_EXECUTE_WARNING(llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop not parallel: ";
                                      I.print(llvm::outs()); llvm::outs() << "\n";)
        return false;
      };
      // The memory instruction can refer to the loop identifier metadata
      // directly or indirectly through another list metadata (in case of
      // nested parallel loops). The loop identifier metadata refers to
      // itself so we can check both cases with the same routine.
      llvm::MDNode *LoopIdMD = I.getMetadata(llvm::LLVMContext::MD_mem_parallel_loop_access);

      if (!LoopIdMD)
        return ReturnFalse();

      if (!llvm::is_contained(LoopIdMD->operands(), DesiredLoopIdMetadata))
        return ReturnFalse();
    }
  }
  return true;
}

/*!
 * In _too simple_ loops, we might not have a dedicated latch.. so make one!
 * Only simple / canonical loops supported.
 *
 * Also adds vectorization hint to latch, so only use for work item loops..
 *
 * @param L The loop without a dedicated latch.
 * @param Latch The loop latch.
 * @param LI LoopInfo to be updated.
 * @param DT DominatorTree to be updated.
 * @return The new latch block, mostly containing the loop induction instruction.
 */
llvm::BasicBlock *simplifyLatch(const llvm::Loop *L, llvm::BasicBlock *Latch, llvm::LoopInfo &LI,
                                llvm::DominatorTree &DT) {
  assert(L->getCanonicalInductionVariable() && "must be canonical loop!");
  llvm::Value *InductionValue = L->getCanonicalInductionVariable()->getIncomingValueForBlock(Latch);
  auto *InductionInstr = llvm::cast<llvm::Instruction>(InductionValue);
  auto *NewLatch = llvm::SplitBlock(Latch, InductionInstr, &DT, &LI, nullptr, Latch->getName() + ".latch");

  // work-item loops should really always be vectorizable, so emit metadata to suggest so
  if (!llvm::findOptionMDForLoop(L, "llvm.loop.vectorize.enable")) {
    llvm::IRBuilder MDBuilder{NewLatch->getContext()};
    auto *MDVectorize = llvm::MDNode::get(NewLatch->getContext(),
                                          {llvm::MDString::get(NewLatch->getContext(), "llvm.loop.vectorize.enable"),
                                           llvm::ConstantAsMetadata::get(MDBuilder.getTrue())});
    auto *LoopID = llvm::makePostTransformationMetadata(NewLatch->getContext(), L->getLoopID(), {}, {MDVectorize});
    L->setLoopID(LoopID);
  }
  return NewLatch;
}

llvm::BasicBlock *splitEdge(llvm::BasicBlock *Root, llvm::BasicBlock *&Target, llvm::LoopInfo *LI,
                            llvm::DominatorTree *DT) {
  auto *NewInnerLoopExitBlock = llvm::SplitEdge(Root, Target, DT, LI, nullptr);
#if LLVM_VERSION_MAJOR < 12
  // NewInnerLoopExitBlock should be between header and InnerLoopExitBlock
  // SplitEdge behaviour was fixed in LLVM 12 to actually ensure this.
  std::swap(NewInnerLoopExitBlock, Target);
#endif
  return NewInnerLoopExitBlock;
}

llvm::Instruction *getBrCmp(const llvm::BasicBlock &BB) {
  if (auto *BI = llvm::dyn_cast_or_null<llvm::BranchInst>(BB.getTerminator()))
    if (BI->isConditional()) {
      if (auto *CmpI = llvm::dyn_cast<llvm::ICmpInst>(BI->getCondition()))
        return CmpI;
      else if (auto *SelectI = llvm::dyn_cast<llvm::SelectInst>(BI->getCondition()))
        return SelectI;
    }
  return nullptr;
}

llvm::SmallPtrSet<llvm::PHINode *, 2> getInductionVariables(const llvm::Loop &L) {
  // adapted from LLVM 11s Loop->getInductionVariable, just finding an induction var in more cases..
  if (!L.isLoopSimplifyForm())
    return {};

  llvm::BasicBlock *Header = L.getHeader();
  assert(Header && "Expected a valid loop header");
  llvm::Instruction *CmpInst = getBrCmp(*Header);
  if (!CmpInst) {
    CmpInst = getBrCmp(*L.getLoopLatch());
    if (!CmpInst)
      return {};
  }

  // check we have at most 2 actual pseudo and %c = select i1 %c1, i1 %c2, i1 false
  llvm::SmallPtrSet<llvm::Instruction *, 2> Cmps;
  for (auto &OP : CmpInst->operands())
    if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OP))
      Cmps.insert(OPI);

  llvm::SmallPtrSet<llvm::PHINode *, 2> IndVars;
  for (llvm::PHINode &IndVar : Header->phis()) {
    HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Header PHI: "; IndVar.print(llvm::outs());
                               llvm::outs() << "\n";)

    // case 1:
    // IndVar = phi[{InitialValue, preheader}, {StepInst, latch}]
    // cmp = IndVar < FinalValue
    // StepInst = IndVar + step
    if (std::find(Cmps.begin(), Cmps.end(), &IndVar) != Cmps.end())
      IndVars.insert(&IndVar);
    // case 2:
    // IndVar = phi[{InitialValue, preheader}, {StepInst, latch}]
    // StepInst = IndVar + step
    // cmp = StepInst < FinalValue
    else if (std::any_of(Cmps.begin(), Cmps.end(), [&IndVar](auto *Cmp) {
               return std::find(Cmp->op_begin(), Cmp->op_end(), &IndVar) != Cmp->op_end();
             }))
      IndVars.insert(&IndVar);
  }

  return IndVars;
}

// only for inner loops required..
llvm::BasicBlock *simplifyLatchNonCanonical(const llvm::Loop *L, llvm::BasicBlock *Latch, llvm::LoopInfo &LI,
                                            llvm::DominatorTree &DT) {
  auto IndVars = getInductionVariables(*L);
  assert(!IndVars.empty() && "Loop ind vars must be found");

  llvm::SmallVector<llvm::PHINode *, 2> IndVarVec{IndVars.begin(), IndVars.end()};
  std::sort(IndVarVec.begin(), IndVarVec.end(),
            [&DT](llvm::PHINode *IndVar, llvm::PHINode *IndVar2) { return !DT.dominates(IndVar, IndVar2); });
  auto *PhiI = IndVarVec[0];
  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Loop ind var: "; PhiI->print(llvm::outs());
                             llvm::outs() << "\n";)
  auto *InductionInstr = llvm::cast<llvm::Instruction>(PhiI->getIncomingValueForBlock(Latch));
  return llvm::SplitBlock(Latch, InductionInstr, &DT, &LI, nullptr, Latch->getName() + ".latch");
}

void moveNonIndVarOutOfHeader(llvm::Loop &L, llvm::Loop &PrevL, llvm::Value *Idx, llvm::MDNode *MDAccessGroup) {
  const auto IndPhis = getInductionVariables(L);
  assert(!IndPhis.empty() && "No Loop induction variable found.");

  auto *Header = L.getHeader();
  llvm::DenseMap<llvm::Value *, llvm::Instruction *> ValueAllocaMap;
  llvm::SmallVector<llvm::Instruction *, 2> ToErase;
  llvm::SmallPtrSet<llvm::Instruction *, 8> DependingInsts;
  for (auto &PhiI : Header->phis()) {
    if (!IndPhis.contains(&PhiI)) {
      llvm::AllocaInst *AllocaI = nullptr;
      auto *FromPreHeaderV = PhiI.getIncomingValueForBlock(L.getLoopPreheader());
      if (auto *FromPreHeaderLI = llvm::dyn_cast<llvm::LoadInst>(FromPreHeaderV)) {
        AllocaI = getLoopStateAllocaForLoad(*FromPreHeaderLI);
      } else { // constant values
        assert(!llvm::isa<llvm::Instruction>(FromPreHeaderV) && "If input not load, then only const allowed");

        auto *IP = llvm::dyn_cast<llvm::Instruction>(PrevL.getLoopLatch()->getFirstNonPHIOrDbgOrLifetime());
        AllocaI = arrayifyValue(Header->getParent()->getEntryBlock().getFirstNonPHIOrDbg(), FromPreHeaderV, IP,
                                PrevL.getCanonicalInductionVariable(), MDAccessGroup);
      }
      if (auto *FromLatchV = PhiI.getIncomingValueForBlock(L.getLoopLatch())) {
        // todo: might need an IP.. if value before first use or something..?
        storeToAlloca(*FromLatchV, *AllocaI, *Idx, MDAccessGroup);
        for (auto *U : PhiI.users())
          if (auto *UInst = llvm::dyn_cast<llvm::Instruction>(U))
            DependingInsts.insert(UInst);
        ToErase.push_back(&PhiI);
        ValueAllocaMap[&PhiI] = AllocaI;
      }
    }
  }
  replaceOperandsWithArrayLoad(Idx, ValueAllocaMap, DependingInsts, MDAccessGroup);
  for (auto *PhiI : ToErase) {
    PhiI->eraseFromParent();
  }
}

void replaceIndexWithNull(const llvm::SmallVectorImpl<llvm::BasicBlock *> &Blocks, llvm::Instruction *IP,
                          const llvm::PHINode *Index) {
  llvm::ValueToValueMapTy VMap;
  VMap[Index] = llvm::Constant::getNullValue(Index->getType());
  llvm::remapInstructionsInBlocks(Blocks, VMap);
}

void replaceIndexWithNull(const llvm::Loop *InnerLoop, const llvm::PHINode *Index) {
  llvm::SmallVector<llvm::BasicBlock *, 8> BBs{InnerLoop->block_begin(), InnerLoop->block_end()};
  BBs.push_back(InnerLoop->getLoopPreheader());
  replaceIndexWithNull(BBs, InnerLoop->getLoopPreheader()->getFirstNonPHI(), Index);
}

llvm::LoadInst *getLoadInBlockForAlloca(const llvm::BasicBlock *SearchBlock, llvm::AllocaInst *Alloca) {
  for (auto *User : Alloca->users()) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Alloca Use: ";
    if (auto *UInst = llvm::dyn_cast<llvm::Instruction>(User)) {
      UInst->print(llvm::outs());
      if (UInst->getParent() == SearchBlock) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(UInst)) {
          auto U =
              std::find_if(GEP->user_begin(), GEP->user_end(), [](auto *U) { return llvm::isa<llvm::LoadInst>(U); });
          if (U != GEP->user_end())
            UInst = llvm::cast<llvm::LoadInst>(*U);
        }
        if (auto *LoadInst = llvm::dyn_cast<llvm::LoadInst>(UInst)) {
          return LoadInst;
        }
      }
    }
    llvm::outs() << "\n";
  }
  return nullptr;
}

void replacePredecessorsSuccessor(llvm::BasicBlock *OldBlock, llvm::BasicBlock *NewBlock, llvm::DominatorTree &DT,
                                  llvm::SmallVectorImpl<llvm::BasicBlock *> *PredsToIgnore = nullptr) {
  llvm::SmallVector<llvm::BasicBlock *, 2> Preds{llvm::pred_begin(OldBlock), llvm::pred_end(OldBlock)};
  llvm::BasicBlock *Ncd = nullptr;
  for (auto *Pred : Preds) {
    std::size_t SuccIdx = 0;
    for (auto *Succ : llvm::successors(Pred)) {
      if (Succ == OldBlock)
        break;
      ++SuccIdx;
    }
    Ncd = Ncd ? DT.findNearestCommonDominator(Ncd, Pred) : Pred;

    if (!PredsToIgnore || std::find(PredsToIgnore->begin(), PredsToIgnore->end(), Pred) == PredsToIgnore->end())
      Pred->getTerminator()->setSuccessor(SuccIdx, NewBlock);
  }

  if (!DT.getNode(NewBlock))
    DT.addNewBlock(NewBlock, Ncd);
  else if (Ncd && DT.getNode(Ncd))
    DT.changeImmediateDominator(NewBlock, Ncd);
}

/// (possible) side effect: replacePredecessorsSuccessor(NewTarget, OldLatch)
void getCondTargets(const llvm::BasicBlock *FirstBlock, const llvm::BasicBlock *InnerCond,
                    llvm::BasicBlock *const *BlocksIt, llvm::BasicBlock *const *BlocksEndIt,
                    const llvm::BasicBlock *Merge, llvm::BasicBlock *OldLatch, llvm::DominatorTree &DT,
                    llvm::BasicBlock *&NewTarget, llvm::BasicBlock *&OldTarget,
                    llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> &CondsAndMerges) {
  if (InnerCond && BlocksIt != BlocksEndIt &&
      (std::find(CondsAndMerges[InnerCond]->BlocksRight.begin(), CondsAndMerges[InnerCond]->BlocksRight.end(),
                 *BlocksIt) != CondsAndMerges[InnerCond]->BlocksRight.end() ||
       std::find(CondsAndMerges[InnerCond]->BlocksLeft.begin(), CondsAndMerges[InnerCond]->BlocksLeft.end(),
                 *BlocksIt) != CondsAndMerges[InnerCond]->BlocksLeft.end())) {
    NewTarget = const_cast<llvm::BasicBlock *>(InnerCond);
    if (BlocksIt == BlocksEndIt || NewTarget == *BlocksIt)
      OldTarget = OldLatch;
  } else if (BlocksIt != BlocksEndIt) {
    NewTarget = *BlocksIt;
    replacePredecessorsSuccessor(NewTarget, OldLatch, DT);
  } else if (FirstBlock == Merge) {
    OldTarget = OldLatch;
  }
}

void cloneConditions(llvm::Function *F,
                     llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> &CondsAndMerges,
                     llvm::SmallVector<llvm::BasicBlock *, 8> &BeforeSplitBlocks,
                     llvm::SmallVector<llvm::BasicBlock *, 8> &AfterSplitBlocks, llvm::Loop &NewLoop,
                     llvm::BasicBlock *NewHeader, llvm::BasicBlock *NewLatch, llvm::BasicBlock *OldLatch,
                     llvm::BasicBlock *&FirstBlockInNew, llvm::BasicBlock *&LastCondInNew, llvm::DominatorTree &DT,
                     llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<const llvm::BasicBlock *, 8> SortedCondBlocks;
  std::transform(CondsAndMerges.begin(), CondsAndMerges.end(), std::back_inserter(SortedCondBlocks),
                 [](auto &Pair) { return Pair.first; });
  std::sort(SortedCondBlocks.begin(), SortedCondBlocks.end(),
            [&DT](auto *FirstBB, auto *SecondBB) { return DT.dominates(FirstBB, SecondBB); });

  for (auto *CondBB : SortedCondBlocks) {
    auto *Cond = CondsAndMerges[CondBB].get();
    if (Cond->Cond == Cond->Merge)
      continue;
    auto *BlocksLeftIt = std::find_if(Cond->BlocksLeft.begin(), Cond->BlocksLeft.end(), [&BeforeSplitBlocks](auto *BB) {
      return std::find(BeforeSplitBlocks.begin(), BeforeSplitBlocks.end(), BB) == BeforeSplitBlocks.end();
    });
    auto *BlocksRightIt =
        std::find_if(Cond->BlocksRight.begin(), Cond->BlocksRight.end(), [&BeforeSplitBlocks](auto *BB) {
          return std::find(BeforeSplitBlocks.begin(), BeforeSplitBlocks.end(), BB) == BeforeSplitBlocks.end();
        });
    if (std::find(BeforeSplitBlocks.begin(), BeforeSplitBlocks.end(), Cond->Cond) != BeforeSplitBlocks.end() &&
        (BlocksLeftIt != Cond->BlocksLeft.end() || BlocksRightIt != Cond->BlocksRight.end())) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << Cond->Cond->getName() << "\n";

      auto *DefaultNewTarget = NewLatch;
      llvm::BasicBlock *DefaultOldTarget = nullptr;

      auto *LeftTarget = DefaultNewTarget;
      auto *OldLeftTarget = DefaultOldTarget;
      getCondTargets(Cond->Cond->getTerminator()->getSuccessor(0), Cond->InnerCondLeft, BlocksLeftIt,
                     Cond->BlocksLeft.end(), Cond->Merge, OldLatch, DT, LeftTarget, OldLeftTarget, CondsAndMerges);

      auto *RightTarget = DefaultNewTarget;
      llvm::BasicBlock *OldRightTarget = DefaultOldTarget;
      getCondTargets(Cond->Cond->getTerminator()->getSuccessor(1), Cond->InnerCondRight, BlocksRightIt,
                     Cond->BlocksRight.end(), Cond->Merge, OldLatch, DT, RightTarget, OldRightTarget, CondsAndMerges);

      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "  left target: " << LeftTarget->getName() << "\n";
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "  right target: " << RightTarget->getName() << "\n";

      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO
                   << "  old left target: " << (OldLeftTarget ? OldLeftTarget->getName() : "") << "\n";
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO
                   << "  old right target: " << (OldRightTarget ? OldRightTarget->getName() : "") << "\n";

      auto *NewCond = llvm::BasicBlock::Create(Cond->Cond->getContext(),
                                               Cond->Cond->getName() + llvm::Twine(".condcopy"), F, nullptr);
      llvm::IRBuilder TermBuilder{NewCond, NewCond->getFirstInsertionPt()};
      auto *BCV = Cond->Cond->getTerminator()->getOperand(0);
      if (auto *BCI = llvm::dyn_cast<llvm::Instruction>(BCV)) {
        auto *NewBCI = BCI->clone();
        VMap[BCI] = NewBCI;
        BCV = NewBCI;
      }
      auto *NewCondBr = TermBuilder.CreateCondBr(BCV, LeftTarget, RightTarget,
                                                 const_cast<llvm::Instruction *>(Cond->Cond->getTerminator()));
      if (BCV != Cond->Cond->getTerminator()->getOperand(0))
        if (auto *NewBCI = llvm::dyn_cast<llvm::Instruction>(BCV))
          NewBCI->insertBefore(NewCond->getFirstNonPHI());

      VMap[Cond->Cond->getTerminator()] = NewCondBr;

      VMap[Cond->Cond] = NewCond;
      NewLoop.addBlockEntry(NewCond);
      AfterSplitBlocks.push_back(NewCond);

      if (!Cond->InnerCondLeft && !Cond->InnerCondRight) {
        LastCondInNew = NewCond;
      }

      if (!Cond->ParentCond) {
        DT.addNewBlock(NewCond, NewHeader);

        llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "   overwriting FirstBlockInNew " << FirstBlockInNew->getName()
                     << " with " << Cond->Cond->getName() << "\n";
        FirstBlockInNew = NewCond;
      } else if (VMap.find(Cond->ParentCond) != VMap.end()) {
        if (auto *NewParent = llvm::dyn_cast<llvm::BasicBlock>(VMap[Cond->ParentCond])) {
          DT.addNewBlock(NewCond, NewParent);
        }
      }
      llvm::outs().flush();

      if (OldLeftTarget) {
        const_cast<llvm::BasicBlock *>(Cond->Cond)->getTerminator()->setSuccessor(0, OldLeftTarget);
      }
      if (OldRightTarget) {
        const_cast<llvm::BasicBlock *>(Cond->Cond)->getTerminator()->setSuccessor(1, OldRightTarget);
      }
    }
  }
}

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup) {
  // findOptionMDForLoopID also checks if there's a loop id, so this is fine
  if (auto *ParAccesses = llvm::findOptionMDForLoopID(L->getLoopID(), "llvm.loop.parallel_accesses")) {
    llvm::SmallVector<llvm::Metadata *, 4> AccessGroups{ParAccesses->op_begin(),
                                                        ParAccesses->op_end()}; // contains .parallel_accesses
    AccessGroups.push_back(MDAccessGroup);
    auto *NewParAccesses = llvm::MDNode::get(F->getContext(), AccessGroups);

    const auto *const PIt = std::find(L->getLoopID()->op_begin(), L->getLoopID()->op_end(), ParAccesses);
    auto PIdx = std::distance(L->getLoopID()->op_begin(), PIt);
    L->getLoopID()->replaceOperandWith(PIdx, NewParAccesses);
  } else {
    auto *NewParAccesses = llvm::MDNode::get(
        F->getContext(), {llvm::MDString::get(F->getContext(), "llvm.loop.parallel_accesses"), MDAccessGroup});
    L->setLoopID(llvm::makePostTransformationMetadata(F->getContext(), L->getLoopID(), {}, {NewParAccesses}));
  }
}

void splitIntoWorkItemLoops(llvm::BasicBlock *LastOldBlock, llvm::BasicBlock *FirstNewBlock,
                            const llvm::BasicBlock *PreHeader, llvm::BasicBlock *Header, llvm::BasicBlock *Latch,
                            llvm::BasicBlock *ExitBlock, llvm::Function *F, llvm::Loop *&L, llvm::Loop *ParentLoop,
                            llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::ScalarEvolution &SE,
                            llvm::AssumptionCache &AC, const std::function<void(llvm::Loop &)> &LoopAdder,
                            const llvm::StringRef &Suffix, llvm::DbgValueInst *ThisDbgValue,
                            llvm::MDNode *MDAccessGroup) {
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG();)

  llvm::ValueToValueMapTy VMap;

  auto CondsAndMerges = findIfCondition(Header, {ExitBlock, Header}, L->getLoopsInPreorder(), DT);
  llvm::SmallVector<llvm::BasicBlock *, 8> BeforeSplitBlocks;
  findBaseBlocks(Header, Latch, {LastOldBlock, ExitBlock}, DT, BeforeSplitBlocks, CondsAndMerges);
  BeforeSplitBlocks.push_back(LastOldBlock);

  llvm::SmallVector<llvm::BasicBlock *, 8> AfterSplitBlocks;
  fillDescendantsExcl(Header, {Latch, ExitBlock}, DT, AfterSplitBlocks);
  AfterSplitBlocks.push_back(Latch); // todo: maybe uniquify..

  AfterSplitBlocks.erase(std::remove_if(AfterSplitBlocks.begin(), AfterSplitBlocks.end(),
                                        [&BeforeSplitBlocks](auto *BB) {
                                          return std::find(BeforeSplitBlocks.begin(), BeforeSplitBlocks.end(), BB) !=
                                                 BeforeSplitBlocks.end();
                                        }),
                         AfterSplitBlocks.end());

  arrayifyAllocas(&F->getEntryBlock(), *L, L->getCanonicalInductionVariable(), DT, MDAccessGroup);

  // remove latch again..
  AfterSplitBlocks.erase(
      std::remove_if(AfterSplitBlocks.begin(), AfterSplitBlocks.end(), [Latch](auto *BB) { return BB == Latch; }),
      AfterSplitBlocks.end());

  llvm::Loop &NewLoop = *LI.AllocateLoop();
  ParentLoop->addChildLoop(&NewLoop);

  VMap[PreHeader] = Header;

  auto *NewHeader = llvm::CloneBasicBlock(Header, VMap, Suffix, F, nullptr, nullptr);
  VMap[Header] = NewHeader;
  NewLoop.addBlockEntry(NewHeader);
  NewLoop.moveToHeader(NewHeader);
  AfterSplitBlocks.push_back(NewHeader);
  dropDebugLocation(NewHeader);

  auto *NewLatch = llvm::CloneBasicBlock(Latch, VMap, Suffix, F, nullptr, nullptr);
  VMap[Latch] = NewLatch;
  NewLoop.addBlockEntry(NewLatch);
  AfterSplitBlocks.push_back(NewLatch);
  dropDebugLocation(NewLatch);

  //  L->removeBlockFromLoop(FirstNewBlock);
  NewLoop.addBlockEntry(FirstNewBlock);

  // connect new loop
  //    NewHeader->getTerminator()->setSuccessor(0, NewBlock);
  NewHeader->getTerminator()->setSuccessor(1, ExitBlock);
  DT.addNewBlock(NewHeader, Header);
  //    DT.changeImmediateDominator(NewBlock, NewHeader);
  DT.changeImmediateDominator(ExitBlock, NewHeader);

  replacePredecessorsSuccessor(Latch, NewLatch, DT);

  NewLatch->getTerminator()->setSuccessor(0, NewHeader);
  //    DT.changeImmediateDominator(newHeader, newLatch);

  // fix old loop
  Header->getTerminator()->setSuccessor(1, NewHeader);

  replacePredecessorsSuccessor(FirstNewBlock, Latch, DT, &AfterSplitBlocks);

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "conds to clone\n";
  llvm::BasicBlock *FirstBlockInNew = FirstNewBlock;
  llvm::BasicBlock *LastCondInNew = NewHeader;
  cloneConditions(F, CondsAndMerges, BeforeSplitBlocks, AfterSplitBlocks, NewLoop, NewHeader, NewLatch, Latch,
                  FirstBlockInNew, LastCondInNew, DT, VMap);
  VMap[LastOldBlock] = LastCondInNew;
  llvm::outs().flush();

  auto *LoadBlock = llvm::BasicBlock::Create(F->getContext(), "loadBB" + Suffix, F, FirstBlockInNew);
  NewHeader->getTerminator()->setSuccessor(0, LoadBlock);
  DT.addNewBlock(LoadBlock, NewHeader);

  {
    llvm::IRBuilder TermBuilder{LoadBlock, LoadBlock->getFirstInsertionPt()};
    TermBuilder.CreateBr(FirstBlockInNew);
    DT.changeImmediateDominator(FirstBlockInNew, LoadBlock);
    FirstBlockInNew->replacePhiUsesWith(LastOldBlock,
                                        LoadBlock); // won't be rewritten by remapping, so need to do manually

    if (ThisDbgValue) { // copy the kernel functor's this pointer debug value to enable referencing it's members.
      llvm::DIBuilder DbgBuilder{*ThisDbgValue->getParent()->getParent()->getParent()};
      DbgBuilder.insertDbgValueIntrinsic(ThisDbgValue->getValue(), ThisDbgValue->getVariable(),
                                         ThisDbgValue->getExpression(), ThisDbgValue->getDebugLoc(),
                                         LoadBlock->getFirstNonPHI());
    }
  }

  arrayifyDependencies(F, L->getCanonicalInductionVariable(), BeforeSplitBlocks, AfterSplitBlocks, LoadBlock, VMap,
                       MDAccessGroup);

  AfterSplitBlocks.push_back(LoadBlock);
  llvm::SmallVector<llvm::BasicBlock *, 8> BbToRemap = AfterSplitBlocks;

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "BLOCKS TO REMAP " << BbToRemap.size() << "\n";
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> BBSet{AfterSplitBlocks.begin(), AfterSplitBlocks.end()};
  for (auto *BB : BBSet)
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << " " << BB->getName() << "\n";
  llvm::outs().flush();

  llvm::remapInstructionsInBlocks(BbToRemap, VMap);
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *Block
                                  : L->getParentLoop()->blocks()) {
    if (!Block->getParent())
      Block->print(llvm::errs());
  } llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO
                 << "new loopx.. " << &NewLoop << " with parent " << NewLoop.getParentLoop() << "\n";)
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(DT.print(llvm::errs());)
  L = updateDtAndLi(LI, DT, NewLatch, *L->getHeader()->getParent());
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(DT.print(llvm::errs());)
  LoopAdder(*L);

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "new loop.. " << L << " with parent " << L->getParentLoop() << "\n";
  llvm::simplifyLoop(L->getParentLoop(), &DT, &LI, &SE, &AC, nullptr, false);
  for (auto *Block : L->blocks()) // need pre-headers -> after simplify
    moveArrayLoadForPhiToIncomingBlock(Block);

  L->getLoopLatch()->getTerminator()->setMetadata(hipsycl::compiler::MDKind::WorkItemLoop,
                                                  llvm::MDNode::get(F->getContext(), {}));

  createParallelAccessesMdOrAddAccessGroup(F, L, MDAccessGroup);

  HIPSYCL_DEBUG_EXECUTE_INFO(
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop id for " << L->getHeader()->getName();
      L->getLoopID()->print(llvm::outs(), F->getParent()); for (auto &MDOp
                                                                : llvm::drop_begin(L->getLoopID()->operands(), 1)) {
        MDOp->print(llvm::outs(), F->getParent());
      } llvm::outs() << "\n";)

  if (llvm::verifyFunction(*F, &llvm::errs())) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_ERROR << "function verification failed\n";
  }
}

llvm::AllocaInst *arrayifyIncomingFromPreheader(llvm::PHINode *Phi, const llvm::Loop *PrevLoop,
                                                const llvm::Loop *InnerLoop, llvm::BasicBlock *WorkItemHeader,
                                                llvm::PHINode *OldWIIdx, llvm::PHINode *NewWIIdx,
                                                llvm::ValueToValueMapTy &VMap, llvm::MDTuple *MDAccessGroup) {
  llvm::Instruction *NewHeaderIP = WorkItemHeader->getFirstNonPHIOrDbg();
  llvm::Value *VInc = Phi->getIncomingValueForBlock(InnerLoop->getLoopPreheader());
  if (auto *LInc = llvm::dyn_cast<llvm::LoadInst>(VInc)) {
    llvm::AllocaInst *OrgAlloca = getLoopStateAllocaForLoad(*LInc);

    // create a new alloca, as the load currently is in a load block in the current WI loop, which will be moved inside
    // the inner loop, another load inside the last work item loop is required. That value is then immediately stored to
    // the newly created alloca, which is used exclusively in for the induction variable.
    // todo: if the value from the alloca is really only used as the initial value of the loop, it would be possible to
    //  re-use that alloca and safe stack storage
    auto *ToStore = loadFromAlloca(OrgAlloca, OldWIIdx, PrevLoop->getLoopLatch()->getFirstNonPHIOrDbgOrLifetime(),
                                   MDAccessGroup, LInc->getName());
    llvm::AllocaInst *IncAlloca = arrayifyInstruction(WorkItemHeader->getParent()->getEntryBlock().getFirstNonPHIOrDbg(), ToStore,
                                    PrevLoop->getCanonicalInductionVariable(), MDAccessGroup);

    auto *NewLoad = loadFromAlloca(IncAlloca, NewWIIdx, NewHeaderIP, MDAccessGroup, LInc->getName());
    copyDgbValues(LInc, NewLoad, NewHeaderIP);
    VMap[LInc] = NewLoad;
    VMap[Phi] = NewLoad;
    auto *Copied = loadFromAlloca(IncAlloca, OldWIIdx, InnerLoop->getLoopPreheader()->getFirstNonPHI(), MDAccessGroup,
                                  LInc->getName());
    llvm::dropDebugUsers(*Phi);

    Phi->replaceUsesOfWith(LInc, Copied);
    return IncAlloca;
  } else if (llvm::isa<llvm::Instruction>(VInc)) {
    // todo: instructions, could do the same as with consts, but not sure why we should ever have a non load
    // instruction incoming..
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_ERROR << "Incoming value not load instruction: ";
    VInc->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
    assert(false && "incoming not load -> can't determine alloca");
  } else { // constants
    auto *IP = PrevLoop->getLoopLatch()->getFirstNonPHIOrDbgOrLifetime();

    llvm::AllocaInst *ValueAlloca = arrayifyValue(WorkItemHeader->getParent()->getEntryBlock().getFirstNonPHIOrDbg(),
                                                  VInc, IP, PrevLoop->getCanonicalInductionVariable(), MDAccessGroup);
    auto *Load = loadFromAlloca(ValueAlloca, NewWIIdx, NewHeaderIP, MDAccessGroup, Phi->getName());
    copyDgbValues(Phi, Load, NewHeaderIP);
    llvm::dropDebugUsers(*Phi);
    VMap[Phi] = Load;

    return ValueAlloca;
  }
  return nullptr;
}

llvm::LoadInst *replaceIncomingFromLatchWithLoad(llvm::PHINode *Phi, const llvm::Loop *InnerLoop, llvm::Value *NewWIIdx,
                                                 llvm::AllocaInst *Alloca, llvm::MDTuple *MDAccessGroup) {
  llvm::Value *IncV = Phi->getIncomingValueForBlock(InnerLoop->getLoopLatch());
  auto *IP = InnerLoop->getLoopLatch()->getTerminator();
  auto *LoadI = loadFromAlloca(Alloca, llvm::Constant::getNullValue(NewWIIdx->getType()), IP, MDAccessGroup,
                               IncV->getName() + "LL");
  Phi->replaceUsesOfWith(IncV, LoadI);
  storeToAlloca(*IncV, *Alloca, *NewWIIdx, MDAccessGroup);
  return LoadI;
}

void arrayifyInnerLoopIndVars(const llvm::Loop *NewLoop, const llvm::Loop *PrevLoop, const llvm::Loop *InnerLoop,
                              llvm::BasicBlock *WorkItemHeader, llvm::MDTuple *MDAccessGroup) {
  llvm::ValueToValueMapTy VMap;
  auto InnerIndVars = getInductionVariables(*InnerLoop);
  llvm::PHINode *OldWIIdx = PrevLoop->getCanonicalInductionVariable();
  llvm::PHINode *NewWIIdx = NewLoop->getCanonicalInductionVariable();

  for (auto *Phi : InnerIndVars) {
    if (auto *AllocaI = arrayifyIncomingFromPreheader(Phi, PrevLoop, InnerLoop, WorkItemHeader, OldWIIdx, NewWIIdx,
                                                      VMap, MDAccessGroup)) {
      replaceIncomingFromLatchWithLoad(Phi, InnerLoop, NewWIIdx, AllocaI, MDAccessGroup);
    } else {
      assert(false && "Incoming value not successfully arrayified.");
    }
  }

  llvm::SmallVector<llvm::BasicBlock *, 4> BBsToRemap;
  std::copy_if(InnerLoop->block_begin(), InnerLoop->block_end(), std::back_inserter(BBsToRemap),
               [&InnerLoop](auto BB) { return BB != InnerLoop->getHeader() && BB != InnerLoop->getLoopLatch(); });
  llvm::remapInstructionsInBlocks(BBsToRemap, VMap);
}

void copyLoadsToPreHeader(llvm::BasicBlock *LoadBlock, llvm::BasicBlock *InnerHeader,
                          llvm::BasicBlock *InnerPreHeader) {
  llvm::SmallVector<llvm::BasicBlock *, 1> LoadBlocks{LoadBlock};
  llvm::SmallVector<llvm::BasicBlock *, 2> Headers{InnerPreHeader, InnerHeader};
  llvm::SmallPtrSet<llvm::Instruction *, 4> DependedUponValues;
  llvm::SmallPtrSet<llvm::Instruction *, 4> DependingInsts;
  findDependenciesBetweenBlocks(LoadBlocks, Headers, DependingInsts, DependedUponValues);
  llvm::ValueToValueMapTy HVMap;
  for (auto *I : DependedUponValues) {
    assert(llvm::isa<llvm::LoadInst>(I) && "Depended upon instruction in Load block **must** be Load");
    auto *IC = llvm::cast<llvm::LoadInst>(I->clone());
    auto *GEP = llvm::cast<llvm::GetElementPtrInst>(IC->getPointerOperand());
    IC->replaceUsesOfWith(GEP, GEP->getPointerOperand()); // 0. element

    IC->insertBefore(InnerPreHeader->getTerminator());
    HVMap[I] = IC;
  }
  llvm::remapInstructionsInBlocks(Headers, HVMap);
}

// so the barrier is inside an inner loop.
// we have to invert inner loop and work item loop, so that the work item loop can be split and both w-i loops are
// in the inner loop. This way the barrier semantics are enforced but the loop iterations are still done.
// The inner loop also has iteration variables, these are stored in loop state allocas. For the iteration count
// calculation, the value at index 0 is used, which should be fine as for the barriers the loop trip count must be
// equal for all work items.
// added difficulty comes from the fact that the loop state allocas need to be written and read from within the
// work item loops. This results in the need to actively move the "inner loops'" iteration variable into the last
// work-item loop and store it into the loop state. From there the 0th element is loaded again in the "inner loop"
// latch.
void splitInnerLoop(llvm::Function *F, llvm::Loop *InnerLoop, llvm::Loop *&L, llvm::LoopInfo &LI,
                    llvm::DominatorTree &DT, llvm::ScalarEvolution &SE, llvm::AssumptionCache &AC,
                    llvm::Loop *&ParentLoop, llvm::BasicBlock *&Header, llvm::BasicBlock *&PreHeader,
                    llvm::BasicBlock *&ExitBlock, llvm::BasicBlock *&Latch,
                    const std::function<void(llvm::Loop &)> &LoopAdder, llvm::DbgValueInst *ThisDebugValue,
                    llvm::MDTuple *MDAccessGroup, const llvm::StringRef &BlockNameSuffix) {
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Barrier is in loop..\n";
  assert(InnerLoop->getLoopPreheader() && "must have preheader");
  auto *InnerHeader = InnerLoop->getHeader();
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "InnerHeader: " << InnerHeader->getName() << "\n";

  // split away everything before the inner loop into its own work item loop
  splitIntoWorkItemLoops(InnerLoop->getLoopPreheader(), InnerLoop->getHeader(), PreHeader, Header, Latch, ExitBlock, F,
                         L, ParentLoop, LI, DT, SE, AC, LoopAdder, BlockNameSuffix, ThisDebugValue, MDAccessGroup);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG();)
  auto *NewPreHeader = L->getLoopPreheader();
  auto *NewHeader = L->getHeader();
  auto *NewLoop = LI.getLoopFor(NewHeader);
  assert(NewLoop == L);
  auto *NewLatch = L->getLoopLatch();
  auto *NewExitBlock = L->getExitBlock();

  InnerLoop = LI.getLoopFor(InnerHeader);

  auto *InnerLoopExitBlock = InnerLoop->getExitBlock();
  InnerHeader = InnerLoop->getHeader();
  llvm::BasicBlock *NewInnerLoopExitBlock = splitEdge(InnerHeader, InnerLoopExitBlock, &LI, &DT);

  NewLatch = simplifyLatch(NewLoop, NewLatch, LI, DT);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(llvm::errs() << "cfgbefore 2nd inversion split\n"; DT.print(llvm::errs());)
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewLatch: " << NewLatch->getName() << "\n";
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewHeader: " << NewHeader->getName() << "\n";
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewExitBlock: " << NewExitBlock->getName() << "\n";

  // split away the inner loop's exit block into its own work item loop
  splitIntoWorkItemLoops(NewInnerLoopExitBlock, InnerLoopExitBlock, NewPreHeader, NewHeader, NewLatch, NewExitBlock, F,
                         NewLoop, ParentLoop, LI, DT, SE, AC, LoopAdder, BlockNameSuffix, ThisDebugValue,
                         MDAccessGroup);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG();)

  InnerLoop = LI.getLoopFor(InnerHeader);
  L = LI.getLoopFor(NewHeader);
  auto *LoadBlock = InnerLoop->getLoopPreheader();

  // Split inner latch in a way that all its contents end up in the last work-item loop.
  // Then find inner loop induction variables and clone gep & load if necessary into workitem loop headers
  // then replace all uses of old induction variable with the load
  // As the induction values are now no longer in the latch, they are stored in the wi loop into allocas and loaded
  // in the inner latch
  {
    auto *InnerPreHeader =
        LoadBlock->splitBasicBlock(LoadBlock->getTerminator(), InnerLoop->getHeader()->getName() + ".pre");

    llvm::BasicBlock *InnerLatch = InnerLoop->getLoopLatch();
    InnerLatch = llvm::SplitBlock(InnerLatch, InnerLatch->getTerminator(), &DT, &LI, nullptr,
                                  InnerLatch->getName() + BlockNameSuffix);

    llvm::Loop *PrevLoop = LI.getLoopFor(Header);
    arrayifyInnerLoopIndVars(L, PrevLoop, InnerLoop, NewHeader, MDAccessGroup);

    // move non induction variables out of header and then replace old loop's indices with 0
    // moveNon.. also generates arrayifications for constant values
    moveNonIndVarOutOfHeader(*InnerLoop, *PrevLoop, L->getCanonicalInductionVariable(), MDAccessGroup);
    replaceIndexWithNull(InnerLoop, PrevLoop->getCanonicalInductionVariable());
    llvm::SmallVector<llvm::BasicBlock *, 2> BBs{{InnerLoop->getHeader(), InnerLoop->getLoopPreheader()}};
    replaceIndexWithNull(BBs, InnerLoop->getLoopPreheader()->getFirstNonPHI(), L->getCanonicalInductionVariable());

    copyLoadsToPreHeader(LoadBlock, InnerHeader, InnerPreHeader);
  }

  // perform loop inversion: the inner loop (pre-)header are moved in front of the WI headers and the inner latch
  // becomes the new exit block of the WI loop.
  auto *InnerLatch = InnerLoop->getLoopLatch();
  replacePredecessorsSuccessor(InnerLatch, NewLatch, DT);

  Header->getTerminator()->setSuccessor(1, InnerLoop->getLoopPreheader());
  auto *InnerBody = InnerHeader->getTerminator()->getSuccessor(0);
  InnerHeader->getTerminator()->setSuccessor(0, NewPreHeader);
  InnerHeader->getTerminator()->setSuccessor(1, NewHeader->getTerminator()->getSuccessor(1));
  NewInnerLoopExitBlock->eraseFromParent();

  LoadBlock->getTerminator()->setSuccessor(0, InnerBody);
  NewHeader->getTerminator()->setSuccessor(0, LoadBlock);
  NewHeader->getTerminator()->setSuccessor(1, InnerLatch);

  llvm::MDNode *InnerMD = llvm::MDNode::get(F->getContext(), {llvm::MDString::get(F->getContext(), "hipSYCL_split")});
  InnerLatch->getTerminator()->setMetadata(hipsycl::compiler::MDKind::InnerLoop, InnerMD);

  PreHeader = NewPreHeader;
  Header = NewHeader;
  Latch = NewLatch;
  ExitBlock = InnerLatch;
  ParentLoop = InnerLoop;
  L = updateDtAndLi(LI, DT, NewHeader, *F);

  llvm::errs() << "last cfg in loop inversion\n";
}

// we heavily rely on having the loop induction variables as
// PHINodes and not as alloca store / loads.
// this is what the Mem2Reg or PromoteMemToReg pass does, for -O0 this is not done, thus we need to do this..
void simplifyO0(llvm::Function *F, llvm::Loop *&L, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                llvm::ScalarEvolution &SE, llvm::AssumptionCache &AC, const llvm::TargetTransformInfo &TTI) {
  promoteAllocas(&F->getEntryBlock(), DT, AC);

  // also for O0 builds only, we need to simplify the CFG, as this merges multiple conditional branches into
  // a single loop header, so that it is muuch easier to work with.
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> LoopHeaders;
  for (auto *SL : LI.getLoopsInPreorder()) {
    if (SL->getHeader())
      LoopHeaders.insert(SL->getHeader());
    if (SL->getLoopPreheader())
      LoopHeaders.insert(SL->getLoopPreheader());
  }

  for (auto *Loop : LI.getTopLevelLoops())
    for (auto *Block : Loop->blocks()) {
      llvm::simplifyCFG(Block, TTI, {}, &LoopHeaders);
    }

  // repair loop simplify form
  L = updateDtAndLi(LI, DT, L->getLoopLatch(), *F);
  llvm::simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);
}

bool splitLoop(llvm::Loop *L, llvm::LoopInfo &LI, const std::function<void(llvm::Loop &)> &LoopAdder,
               const llvm::LoopAccessInfo &LAI, llvm::DominatorTree &DT, llvm::ScalarEvolution &SE,
               const llvm::TargetTransformInfo &TTI, llvm::TargetLibraryInfo &TLI,
               const hipsycl::compiler::SplitterAnnotationInfo &SAA, bool IsO0) {
  if (!SAA.isKernelFunc(L->getHeader()->getParent())) {
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

  if (IsO0)
    simplifyO0(F, L, LI, DT, SE, AC, TTI);

  auto *ThisDebugValue = getThisPtrDbgValue(F);
  L->getLoopLatch()->getTerminator()->setMetadata(hipsycl::compiler::MDKind::WorkItemLoop,
                                                  llvm::MDNode::get(F->getContext(), {}));

  auto *MDAccessGroup = llvm::MDNode::getDistinct(F->getContext(), {});
  createParallelAccessesMdOrAddAccessGroup(F, L, MDAccessGroup);
  for (auto *BB : L->blocks()) {
    for (auto &I : *BB) {
      if (I.mayReadOrWriteMemory() && !I.hasMetadata(llvm::LLVMContext::MD_access_group)) {
        addAccessGroupMD(&I, MDAccessGroup);
      }
    }
  }

  if (isAnnotatedParallel(L))
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "loop is parallel\n";
  else if (L->getLoopID()) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop id for " << L->getHeader()->getName();
    L->getLoopID()->print(llvm::outs(), F->getParent());
    for (auto &MDOp : llvm::drop_begin(L->getLoopID()->operands(), 1)) {
      MDOp->print(llvm::outs(), F->getParent());
    }
    llvm::outs() << "\n";
  }

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->print(llvm::outs());)
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG();)

  std::size_t BC = 0;
  for (auto *BarrierIt = Barriers.begin(); BarrierIt != Barriers.end(); ++BarrierIt) {
    auto *Barrier = *BarrierIt;
    Changed = true;
    ++BC;

    auto *BarrierBlock = Barrier->getParent();
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found barrier block: " << BarrierBlock->getName() << "\n";
    L = LI.getLoopFor(BarrierBlock);

    // get work item loop and build up list of inner loops that need to be taken care of
    llvm::SmallVector<llvm::Loop *, 4> InnerLoopWL;
    while (L->getLoopDepth() > 2 &&
           !L->getLoopLatch()->getTerminator()->hasMetadata(hipsycl::compiler::MDKind::WorkItemLoop)) {
      InnerLoopWL.push_back(L);
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "ILP Header: " << L->getHeader()->getName()
                   << " depth: " << L->getLoopDepth() << " parent: " << L->getParentLoop()->getHeader()->getName()
                   << "\n";
      L = L->getParentLoop();
    }

    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found header: " << L->getHeader()->getName() << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO
                 << "Found pre-header: " << (L->getLoopPreheader() ? L->getLoopPreheader()->getName() : "") << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found exit block: " << L->getExitBlock()->getName() << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found latch block: " << L->getLoopLatch()->getName() << "\n";

    llvm::outs().flush();

    llvm::Loop *ParentLoop = L->getParentLoop();
    llvm::BasicBlock *Header = L->getHeader();
    llvm::BasicBlock *PreHeader = L->getLoopPreheader();
    llvm::BasicBlock *ExitBlock = L->getExitBlock();
    llvm::BasicBlock *Latch = L->getLoopLatch();
    Latch = simplifyLatch(L, Latch, LI, DT);

    createParallelAccessesMdOrAddAccessGroup(F, L, MDAccessGroup);

    if (isInConditional(Barrier, DT, Latch)) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "is in conditional\n";
    }

    // need to take care of outer most loop first, so reverse and as LI is changed in a split, the loops become invalid
    // so just store the header from which the correct loop can be determined again.
    llvm::SmallVector<llvm::BasicBlock *, 4> InnerLoopHeaders;
    std::transform(InnerLoopWL.rbegin(), InnerLoopWL.rend(), std::back_inserter(InnerLoopHeaders),
                   [](auto *InnerLoop) { return InnerLoop->getHeader(); });
    for (auto *InnerLoopHeader : InnerLoopHeaders) {
      const std::string BlockNameSuffix = "lsplit" + std::to_string(BC);

      splitInnerLoop(F, LI.getLoopFor(InnerLoopHeader), L, LI, DT, SE, AC, ParentLoop, Header, PreHeader, ExitBlock,
                     Latch, LoopAdder, ThisDebugValue, MDAccessGroup, BlockNameSuffix);
    }

    const std::string BlockNameSuffix = "split" + std::to_string(BC);
    auto *NewBlock =
        llvm::SplitBlock(BarrierBlock, Barrier, &DT, &LI, nullptr, BarrierBlock->getName() + BlockNameSuffix);
    Barrier->eraseFromParent();

    splitIntoWorkItemLoops(BarrierBlock, NewBlock, PreHeader, Header, Latch, ExitBlock, F, L, ParentLoop, LI, DT, SE,
                           AC, LoopAdder, BlockNameSuffix, ThisDebugValue, MDAccessGroup);

    HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG();)
  }

  // get last work item loop, should be fine, as we always split something away after inner loops
  while (L->getLoopDepth() > 1)
    L = L->getParentLoop();
  L = L->getSubLoops().back();
  if (Changed)
    insertLifetimeEndForAllocas(F, L->getCanonicalInductionVariable(), L->getLoopLatch()->getFirstNonPHIOrDbg(),
                                MDAccessGroup);

  //  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG(); F->print(llvm::outs());)
  //  return Changed;
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> LoopHeaders;
  for (auto *SL : LI.getLoopsInPreorder()) {
    if (SL->getHeader())
      LoopHeaders.insert(SL->getHeader());
    if (SL->getLoopPreheader())
      LoopHeaders.insert(SL->getLoopPreheader());
    if (auto *SLatch = SL->getLoopLatch()) {
      if (SLatch->getTerminator()->hasMetadata(llvm::LLVMContext::MD_loop)) {
        llvm::errs() << SLatch->getName() << " ";
        SLatch->getTerminator()->getMetadata(llvm::LLVMContext::MD_loop)->print(llvm::errs());
        llvm::errs() << "\n";
      }
    }
    HIPSYCL_DEBUG_EXECUTE_WARNING(
        if (SL->getLoopLatch()->getTerminator()->hasMetadata(hipsycl::compiler::MDKind::WorkItemLoop)) {
          if (isAnnotatedParallel(SL)) {
            llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "loop is parallel\n";
          } else {
            if (SL->getLoopID()) {
              llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop id for " << SL->getHeader()->getName();
              SL->getLoopID()->print(llvm::outs(), F->getParent());
              for (auto &MDOp : llvm::drop_begin(SL->getLoopID()->operands(), 1)) {
                MDOp->print(llvm::outs(), F->getParent());
              }
              llvm::outs() << "\n";
            }
          }
        })
  }

  for (auto *Loop : LI.getTopLevelLoops())
    for (auto *Block : Loop->blocks()) {
      llvm::SimplifyInstructionsInBlock(Block);
      llvm::simplifyCFG(Block, TTI, {}, &LoopHeaders);
    }

  while (L->getLoopDepth() > 1)
    L = L->getParentLoop();

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << L->getHeader()->getName() << " for inlining:\n";
  //  Changed |= inlineCallsInLoop(L, LI, DT);

  L = updateDtAndLi(LI, DT, L->getHeader(), *L->getHeader()->getParent());

  {
    for (auto *SL : L->getLoopsInPreorder()) {
      if (SL->getLoopLatch()->getTerminator()->hasMetadata(hipsycl::compiler::MDKind::WorkItemLoop)) {
        createParallelAccessesMdOrAddAccessGroup(F, SL, MDAccessGroup);
        for (auto *BB : SL->blocks()) {
          for (auto &I : *BB) {
            if (I.mayReadOrWriteMemory() && !I.hasMetadata(llvm::LLVMContext::MD_access_group)) {
              addAccessGroupMD(&I, MDAccessGroup);
            }
          }
        }
      }
    }
  }

  if (HIPSYCL_DEBUG_LEVEL_INFO <= ::hipsycl::common::output_stream::get().get_debug_level()) {
    for (auto *SL : L->getLoopsInPreorder()) {
      if (SL->getLoopLatch()->getTerminator()->hasMetadata(hipsycl::compiler::MDKind::WorkItemLoop)) {
        if (isAnnotatedParallel(SL))
          llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "loop is parallel\n";
        else if (SL->getLoopID()) {
          assert(SL->getLoopID());
          llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop id for " << SL->getHeader()->getName();
          SL->getLoopID()->print(llvm::outs(), F->getParent());
          for (auto &MDOp : llvm::drop_begin(SL->getLoopID()->operands(), 1)) {
            MDOp->print(llvm::outs(), F->getParent());
          }
          llvm::outs() << "\n";
        }
      }
    }
  }
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F->viewCFG(); F->print(llvm::outs());)
  return Changed;
}

} // namespace

bool hipsycl::compiler::LoopSplitAtBarrierPassLegacy::runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) {
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<llvm::AAResultsWrapperPass>();

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &SE = getAnalysis<llvm::ScalarEvolutionWrapperPass>().getSE();
  const auto &TTI = getAnalysis<llvm::TargetTransformInfoWrapperPass>().getTTI(*L->getHeader()->getParent());
  auto &TLI = getAnalysis<llvm::TargetLibraryInfoWrapperPass>().getTLI(*L->getHeader()->getParent());
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  llvm::LoopAccessInfo LAI(L, &SE, &TLI, &AA.getAAResults(), &DT, &LI);
  return splitLoop(
      L, LI, [&LPM](llvm::Loop &L) { LPM.addLoop(L); }, LAI, DT, SE, TTI, TLI, SAA, IsO0_);
}

void hipsycl::compiler::LoopSplitAtBarrierPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::AAResultsWrapperPass>();
  AU.addPreserved<llvm::AAResultsWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::TargetTransformInfoWrapperPass>();
  AU.addPreserved<llvm::TargetTransformInfoWrapperPass>();
  AU.addRequired<llvm::TargetLibraryInfoWrapperPass>();
  AU.addPreserved<llvm::TargetLibraryInfoWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses hipsycl::compiler::LoopSplitAtBarrierPass::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                                                                       llvm::LoopStandardAnalysisResults &AR,
                                                                       llvm::LPMUpdater &LPMU) {
  auto &SAA = AM.getResult<SplitterAnnotationAnalysis>(L, AR);
  const auto &LAI = AM.getResult<llvm::LoopAccessAnalysis>(L, AR);
  const auto &TTI = AM.getResult<llvm::TargetIRAnalysis>(L, AR);
  auto &TLI = AM.getResult<llvm::TargetLibraryAnalysis>(L, AR);
  if (!splitLoop(
          &L, AR.LI, [&LPMU](llvm::Loop &L) { LPMU.addSiblingLoops({&L}); }, LAI, AR.DT, AR.SE, TTI, TLI, SAA, IsO0_))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA = llvm::getLoopPassPreservedAnalyses();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  PA.preserve<llvm::AAManager>();
  PA.preserve<llvm::TargetIRAnalysis>();
  return PA;
}

char hipsycl::compiler::SplitterAnnotationAnalysisLegacy::ID = 0;
char hipsycl::compiler::LoopSplitAtBarrierPassLegacy::ID = 0;
llvm::AnalysisKey hipsycl::compiler::SplitterAnnotationAnalysis::Key;
