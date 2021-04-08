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

namespace hipsycl {
namespace compiler {
static constexpr size_t NumArrayElements = 1024;
static constexpr const char MetadataKind[] = "hipSYCL";
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

hipsycl::compiler::SplitterAnnotationInfo::SplitterAnnotationInfo(const llvm::Module &Module) { analyzeModule(Module); }

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  if (SplitterAnnotation)
    return false;
  SplitterAnnotation = SplitterAnnotationInfo{*F.getParent()};
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
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << N->getBlock()->getName() << "\n";
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
    const std::vector<llvm::Loop *> &Loops, const llvm::DominatorTree &DT,
    llvm::SmallPtrSetImpl<const llvm::BasicBlock *> &LookedAtBBs, llvm::SmallVectorImpl<Condition *> &BranchStack,
    llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> &BranchCondAndMerge) {
  llvm::BasicBlock *IfThen = nullptr, *IfElse = nullptr;

  const auto NumSuccessors = Root->getTerminator()->getNumSuccessors();
  if (NumSuccessors > 1) {
    auto LoopIt = std::find_if(Loops.begin(), Loops.end(), [Root](auto *Loop) { return Loop->getHeader() == Root; });
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
      if (LookedAtBBs.insert(Successor).second) {
        findIfConditionInner(Successor, Terminals, Loops, DT, LookedAtBBs, BranchStack, BranchCondAndMerge);
      } else if (Successor->hasNPredecessorsOrMore(2) &&
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
    }
  }
}

llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8>
findIfCondition(const llvm::BasicBlock *Root, const llvm::ArrayRef<const llvm::BasicBlock *> Terminals,
                const std::vector<llvm::Loop *> &Loops, const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> LookedAtBBs;
  llvm::SmallVector<Condition *, 8> BranchStack;
  llvm::SmallDenseMap<const llvm::BasicBlock *, std::unique_ptr<Condition>, 8> BranchCondAndMerge;

  LookedAtBBs.insert(Root);
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
          DependingInsts.insert(I);
          DependedUponValues.insert(V);
        }
      }
    }
  }
}

void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::Value *Idx, const llvm::DominatorTree &DT,
                     llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap) {
  auto *MDAlloca =
      llvm::MDNode::get(EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), "hipSYCLLoopState")});
  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (llvm::MDNode *MD = Alloca->getMetadata(hipsycl::compiler::MetadataKind))
        continue; // already arrayificated
      if (Alloca->getName().startswith(".omp.") || Alloca->getName().startswith("barrier") ||
          Alloca->getName().startswith("group_id"))
        continue; // todo: replace with dependency analysis alla fillDependingInsts
      WL.push_back(Alloca);
    }
  }

  for (auto *I : WL) {
    llvm::IRBuilder AllocaBuidler{I};
    llvm::Type *T = I->getAllocatedType();
    if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
      auto ArrSize = ArrSizeC->getLimitedValue();
      if (ArrSize > 1) {
        T = llvm::ArrayType::get(T, ArrSize);
        llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Caution, alloca was array\n";
      }
    }

    auto *Alloca = AllocaBuidler.CreateAlloca(T, AllocaBuidler.getInt32(hipsycl::compiler::NumArrayElements),
                                              I->getName() + "_alloca");
    Alloca->setMetadata(hipsycl::compiler::MetadataKind, MDAlloca);
    ValueAllocaMap[I] = Alloca;

    llvm::Instruction *GepIp = nullptr;
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!GepIp)
          GepIp = UI;
        else if (DT.dominates(UI, GepIp)) {
          GepIp = UI;
        }
      }
    }
    if (GepIp) {
      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEPV = LoadBuilder.CreateGEP(Alloca, Idx, I->getName() + "_gep");
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(GEPV);
      GEP->setMetadata(hipsycl::compiler::MetadataKind, MDAlloca);

      I->replaceAllUsesWith(GEP);
      I->eraseFromParent();
    }
  }
}

void arrayifyDependedUponValues(llvm::Instruction *IPAllocas, llvm::Value *Idx,
                                const llvm::SmallPtrSet<llvm::Instruction *, 8> &DependedUponValues,
                                llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap) {
  auto *MDAlloca =
      llvm::MDNode::get(IPAllocas->getContext(), {llvm::MDString::get(IPAllocas->getContext(), "hipSYCLLoopState")});
  for (auto *I : DependedUponValues) {
    if (auto *MD = I->getMetadata(hipsycl::compiler::MetadataKind))
      continue; // currently just have one MD, so no further value checks

    auto *T = I->getType();
    llvm::IRBuilder AllocaBuilder{IPAllocas};
    auto *Alloca = AllocaBuilder.CreateAlloca(T, AllocaBuilder.getInt32(hipsycl::compiler::NumArrayElements),
                                              I->getName() + "_alloca");
    Alloca->setMetadata(hipsycl::compiler::MetadataKind, MDAlloca);
    ValueAllocaMap[I] = Alloca;

    llvm::IRBuilder WriteBuilder{&*(++I->getIterator())};
    auto *GEP = WriteBuilder.CreateGEP(Alloca, Idx, I->getName() + "_gep");
    WriteBuilder.CreateLifetimeStart(GEP);
    WriteBuilder.CreateStore(I, GEP);
  }
}

llvm::AllocaInst *findAlloca(llvm::Instruction *I) {
  for (auto &OP : I->operands()) {
    if (auto *OPI = llvm::dyn_cast<llvm::AllocaInst>(OP.get()))
      return OPI;
  }

  return nullptr;
}

void replaceOperandsWithArrayLoad(llvm::Value *Idx,
                                  const llvm::DenseMap<llvm::Value *, llvm::Instruction *> &ValueAllocaMap,
                                  const llvm::SmallPtrSet<llvm::Instruction *, 8> &DependingInsts) {
  for (auto *I : DependingInsts) {
    for (auto &OP : I->operands()) {
      auto *OPV = OP.get();
      if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV)) {
        if (auto *MD = OPI->getMetadata(hipsycl::compiler::MetadataKind)) {
          llvm::Instruction *ClonedI = OPI->clone();
          ClonedI->insertBefore(I);
          I->replaceUsesOfWith(OPI, ClonedI); // todo: optimize location and re-usage..
          continue;
        }
      }
      if (auto AllocaIt = ValueAllocaMap.find(OPV); AllocaIt != ValueAllocaMap.end()) {
        auto *Alloca = AllocaIt->getSecond();
        auto LoadBuilder = llvm::IRBuilder(I);
        auto *GEP = LoadBuilder.CreateGEP(Alloca, Idx, OPV->getName() + "_lgep");
        auto *Load = LoadBuilder.CreateLoad(GEP, OPV->getName() + "_load");
        // LoadBuilder.CreateLifetimeEnd(GEP); // todo: at some point we really should care about alloca lifetime

        I->setOperand(OP.getOperandNo(), Load);
      }
    }
  }
}

void arrayifyDependencies(llvm::Function *F, const llvm::Loop *L,
                          llvm::SmallVectorImpl<llvm::BasicBlock *> &ArrfBaseBlocks,
                          llvm::SmallVectorImpl<llvm::BasicBlock *> &ArrfSearchBlocks) {
  llvm::SmallPtrSet<llvm::Instruction *, 8> ArrfDependingInsts;
  llvm::SmallPtrSet<llvm::Instruction *, 8> ArrfDependedUponValues;
  llvm::DenseMap<llvm::Value *, llvm::Instruction *> ValueAllocaMap;

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "baseblocks:\n";
  for (auto *BB : ArrfBaseBlocks) {
    BB->print(llvm::outs());
  }

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "searchblocks:\n";
  for (auto *BB : ArrfSearchBlocks) {
    BB->print(llvm::outs());
  }

  findDependenciesBetweenBlocks(ArrfBaseBlocks, ArrfSearchBlocks, ArrfDependingInsts, ArrfDependedUponValues);
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "depended upon values\n";
  for (auto *V : ArrfDependedUponValues) {
    V->print(llvm::outs());
    llvm::outs() << "\n";
  }
  arrayifyDependedUponValues(F->getEntryBlock().getFirstNonPHI(), L->getCanonicalInductionVariable(),
                             ArrfDependedUponValues, ValueAllocaMap);
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "depending insts\n";
  for (auto *I : ArrfDependingInsts) {
    I->print(llvm::outs());
    llvm::outs() << "\n";
  }
  replaceOperandsWithArrayLoad(L->getCanonicalInductionVariable(), ValueAllocaMap, ArrfDependingInsts);
}

/*!
 * In _too simple_ loops, we might not have a dedicated latch.. so make one!
 * Only simple / canonical loops supported.
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
  return llvm::SplitBlock(Latch, InductionInstr, &DT, &LI, nullptr, Latch->getName() + ".latch");
}
llvm::BasicBlock *simplifyLatchNonCanonical(const llvm::Loop *L, llvm::BasicBlock *Latch, llvm::LoopInfo &LI,
                                            llvm::DominatorTree &DT) {
  assert(&*(++L->getHeader()->begin()) == L->getHeader()->getFirstNonPHI() && "just a single phi compatible for now");
  if (auto *PhiI = llvm::dyn_cast<llvm::PHINode>(L->getHeader()->begin())) {
    auto *InductionInstr = llvm::cast<llvm::Instruction>(PhiI->getIncomingValueForBlock(Latch));
    return llvm::SplitBlock(Latch, InductionInstr, &DT, &LI, nullptr, Latch->getName() + ".latch");
  } else {
    llvm::errs() << "not a phi in loop header\n";
    llvm::errs().flush();
    std::terminate();
  }
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
  for (auto &CondPair : CondsAndMerges) {
    auto *Cond = CondPair.second.get();
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

void splitIntoWorkItemLoops(llvm::BasicBlock *LastOldBlock, llvm::BasicBlock *FirstNewBlock,
                            const llvm::BasicBlock *PreHeader, llvm::BasicBlock *Header, llvm::BasicBlock *Latch,
                            llvm::BasicBlock *ExitBlock, llvm::Function *F, llvm::Loop *&L, llvm::Loop *ParentLoop,
                            llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::ScalarEvolution &SE,
                            llvm::AssumptionCache &AC, const std::function<void(llvm::Loop &)> &LoopAdder,
                            const std::string &Suffix) {
  F->viewCFG();

  llvm::ValueToValueMapTy VMap;

  auto CondsAndMerges = findIfCondition(Header, {ExitBlock, Header}, L->getSubLoops(), DT);
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
  {
    llvm::DenseMap<llvm::Value *, llvm::Instruction *> ValueAllocaMap;
    arrayifyAllocas(&F->getEntryBlock(), L->getCanonicalInductionVariable(), DT, ValueAllocaMap);
  }

  // remove latch again..
  AfterSplitBlocks.erase(
      std::remove_if(AfterSplitBlocks.begin(), AfterSplitBlocks.end(), [Latch](auto *BB) { return BB == Latch; }),
      AfterSplitBlocks.end());

  llvm::Loop &NewLoop = *LI.AllocateLoop();
  ParentLoop->addChildLoop(&NewLoop);

  VMap[PreHeader] = Header;

  llvm::ClonedCodeInfo ClonedCodeInfo;
  auto *NewHeader = llvm::CloneBasicBlock(Header, VMap, Suffix, F, &ClonedCodeInfo, nullptr);
  VMap[Header] = NewHeader;
  NewLoop.addBlockEntry(NewHeader);
  NewLoop.moveToHeader(NewHeader);
  AfterSplitBlocks.push_back(NewHeader);

  auto *NewLatch = llvm::CloneBasicBlock(Latch, VMap, Suffix, F, &ClonedCodeInfo, nullptr);
  VMap[Latch] = NewLatch;
  NewLoop.addBlockEntry(NewLatch);
  AfterSplitBlocks.push_back(NewLatch);

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
  NewHeader->getTerminator()->setSuccessor(0, FirstBlockInNew);
  DT.changeImmediateDominator(FirstBlockInNew, NewHeader);

  arrayifyDependencies(F, L, BeforeSplitBlocks, AfterSplitBlocks);

  llvm::SmallVector<llvm::BasicBlock *, 8> BbToRemap = AfterSplitBlocks;
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "BLOCKS TO REMAP " << BbToRemap.size() << "\n";
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> BBSet{AfterSplitBlocks.begin(), AfterSplitBlocks.end()};

  for (auto *BB : BBSet)
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << " " << BB->getName() << "\n";
  llvm::outs().flush();

  llvm::remapInstructionsInBlocks(BbToRemap, VMap);

  for (auto *Block : L->getParentLoop()->blocks()) {
    if (!Block->getParent())
      Block->print(llvm::errs());
  }
  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "new loopx.. " << &NewLoop << " with parent " << NewLoop.getParentLoop()
               << "\n";
  DT.print(llvm::errs());
  L = updateDtAndLi(LI, DT, NewLatch, *L->getHeader()->getParent());
  DT.print(llvm::errs());
  LoopAdder(*L);

  llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "new loop.. " << L << " with parent " << L->getParentLoop() << "\n";

  for (auto *Block : L->getParentLoop()->blocks()) {
    llvm::SimplifyInstructionsInBlock(Block);
  }

  llvm::simplifyLoop(L->getParentLoop(), &DT, &LI, &SE, &AC, nullptr, false);

  if (llvm::verifyFunction(*F, &llvm::errs())) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "function verification failed\n";
  }
}
bool splitLoop(llvm::Loop *L, llvm::LoopInfo &LI, const std::function<void(llvm::Loop &)> &LoopAdder,
               const llvm::LoopAccessInfo &LAI, llvm::DominatorTree &DT, llvm::ScalarEvolution &SE,
               const llvm::TargetTransformInfo &TTI, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
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

  for (auto *BarrierIt = Barriers.begin(); BarrierIt != Barriers.end(); ++BarrierIt) {
    auto *Barrier = *BarrierIt;
    Changed = true;
    ++BC;

    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found splitter at " << Barrier->getCalledFunction()->getName()
                 << "\n";

    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found header: " << L->getHeader()->getName() << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO
                 << "Found pre-header: " << (L->getLoopPreheader() ? L->getLoopPreheader()->getName() : "") << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found exit block: " << L->getExitBlock()->getName() << "\n";
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found latch block: " << L->getLoopLatch()->getName() << "\n";

    auto *BarrierBlock = Barrier->getParent();
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "Found barrier block: " << BarrierBlock->getName() << "\n";

    llvm::outs().flush();

    llvm::Loop *ParentLoop = L->getParentLoop();
    llvm::BasicBlock *Header = L->getHeader();
    llvm::BasicBlock *PreHeader = L->getLoopPreheader();
    llvm::BasicBlock *ExitBlock = L->getExitBlock();
    llvm::BasicBlock *Latch = L->getLoopLatch();
    Latch = simplifyLatch(L, Latch, LI, DT);

    bool InConditional = isInConditional(Barrier, DT, Latch);
    if (InConditional) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "is in conditional\n";
    }

    if (auto *BarrierLoop = LI.getLoopFor(BarrierBlock); BarrierLoop != L) {
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Barrier is in loop..\n";
      assert(BarrierLoop->getLoopPreheader() && "must have preheader");
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "BHeader0: " << BarrierLoop->getHeader()->getName() << "\n";
      const std::string BlockNameSuffix = "lsplit" + std::to_string(BC);
      splitIntoWorkItemLoops(BarrierLoop->getLoopPreheader(), BarrierLoop->getHeader(), PreHeader, Header, Latch,
                             ExitBlock, F, L, ParentLoop, LI, DT, SE, AC, LoopAdder, BlockNameSuffix);
      F->viewCFG();
      auto *NewPreHeader = Header->getTerminator()->getSuccessor(1);
      auto *NewHeader = NewPreHeader->getTerminator()->getSuccessor(0); // fixme: probably broken
      auto *NewLoop = LI.getLoopFor(NewHeader);
      auto *NewLatch = NewLoop->getLoopLatch();
      auto *NewExitBlock = NewLoop->getExitBlock();

      auto *BarrierLoopExitBlock = BarrierLoop->getExitBlock();
      auto *NewBarrierLoopExitBlock = llvm::BasicBlock::Create(
          NewLatch->getContext(), BarrierLoopExitBlock->getName() + "h" + BlockNameSuffix, F, BarrierLoopExitBlock);
      llvm::BranchInst::Create(BarrierLoopExitBlock, NewBarrierLoopExitBlock);
      BarrierLoop->getHeader()->getTerminator()->setSuccessor(1, NewBarrierLoopExitBlock);
      NewBarrierLoopExitBlock->print(llvm::outs());
      DT.addNewBlock(NewBarrierLoopExitBlock, BarrierLoop->getHeader());
      DT.changeImmediateDominator(BarrierLoopExitBlock, NewBarrierLoopExitBlock);

      NewLatch = simplifyLatch(NewLoop, NewLatch, LI, DT);
      llvm::errs() << "cfgbefore 2nd inversion split\n";
      F->viewCFG();
      DT.print(llvm::errs());
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewLatch: " << NewLatch->getName() << "\n";
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewHeader: " << NewHeader->getName() << "\n";
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "NewExitBlock: " << NewExitBlock->getName() << "\n";

      splitIntoWorkItemLoops(NewBarrierLoopExitBlock, BarrierLoopExitBlock, NewPreHeader, NewHeader, NewLatch,
                             NewExitBlock, F, NewLoop, ParentLoop, LI, DT, SE, AC, LoopAdder, BlockNameSuffix);
      F->viewCFG();
      auto *BHeader = BarrierLoop->getHeader();
      llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "BHeader: " << BHeader->getName() << "\n";

      auto *BLatch = simplifyLatchNonCanonical(BarrierLoop, BarrierLoop->getLoopLatch(), LI, DT);
      replacePredecessorsSuccessor(BLatch, NewLatch, DT);

      Header->getTerminator()->setSuccessor(1, BarrierLoop->getLoopPreheader());
      auto *BBody = BHeader->getTerminator()->getSuccessor(0);
      BHeader->getTerminator()->setSuccessor(0, NewPreHeader);
      BHeader->getTerminator()->setSuccessor(1, NewHeader->getTerminator()->getSuccessor(1));
      NewBarrierLoopExitBlock->eraseFromParent();

      NewHeader->getTerminator()->setSuccessor(0, BBody);
      NewHeader->getTerminator()->setSuccessor(1, BLatch);

      PreHeader = NewPreHeader;
      Header = NewHeader;
      Latch = NewLatch;
      ExitBlock = BLatch;
      ParentLoop = BarrierLoop;
      L = updateDtAndLi(LI, DT, NewHeader, *F);

      llvm::errs() << "last cfg in loop inversion\n";
      F->viewCFG();
    }

    const std::string BlockNameSuffix = "split" + std::to_string(BC);
    auto *NewBlock =
        llvm::SplitBlock(BarrierBlock, Barrier, &DT, &LI, nullptr, BarrierBlock->getName() + BlockNameSuffix);
    Barrier->eraseFromParent();

    splitIntoWorkItemLoops(BarrierBlock, NewBlock, PreHeader, Header, Latch, ExitBlock, F, L, ParentLoop, LI, DT, SE,
                           AC, LoopAdder, BlockNameSuffix);
  }

  llvm::SmallPtrSet<llvm::BasicBlock *, 8> LoopHeaders;
  for (auto *SL : L->getSubLoops()) {
    if (SL->getHeader())
      LoopHeaders.insert(SL->getHeader());
    if (SL->getLoopPreheader())
      LoopHeaders.insert(SL->getLoopPreheader());
  }

  for (auto *Block : L->getParentLoop()->blocks()) {
    llvm::simplifyCFG(Block, TTI, {}, &LoopHeaders);
  }
  L = updateDtAndLi(LI, DT, L->getHeader(), *L->getHeader()->getParent());

  F->viewCFG();
  F->print(llvm::outs());
  return Changed;
}

} // namespace

bool hipsycl::compiler::LoopSplitAtBarrierPassLegacy::runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) {
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<llvm::AAResultsWrapperPass>();

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &SE = getAnalysis<llvm::ScalarEvolutionWrapperPass>().getSE();
  const auto &TTI = getAnalysis<llvm::TargetTransformInfoWrapperPass>().getTTI(*L->getHeader()->getParent());
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  llvm::LoopAccessInfo LAI(L, &SE, nullptr, &AA.getAAResults(), &DT, &LI);
  return splitLoop(
      L, LI, [&LPM](llvm::Loop &L) { LPM.addLoop(L); }, LAI, DT, SE, TTI, SAA);
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

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses hipsycl::compiler::LoopSplitAtBarrierPass::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                                                                       llvm::LoopStandardAnalysisResults &AR,
                                                                       llvm::LPMUpdater &LPMU) {
  auto &SAA = AM.getResult<SplitterAnnotationAnalysis>(L, AR);
  const auto &LAI = AM.getResult<llvm::LoopAccessAnalysis>(L, AR);
  const auto &TTI = AM.getResult<llvm::TargetIRAnalysis>(L, AR);
  if (!splitLoop(
          &L, AR.LI, [&LPMU](llvm::Loop &L) { LPMU.addSiblingLoops({&L}); }, LAI, AR.DT, AR.SE, TTI, SAA))
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
