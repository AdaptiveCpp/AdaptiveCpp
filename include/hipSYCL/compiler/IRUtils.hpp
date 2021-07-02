//
// Created by joachim on 04.06.21.
//

#ifndef HIPSYCL_IRUTILS_HPP
#define HIPSYCL_IRUTILS_HPP

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/LoopInfo.h>

namespace llvm {
class Region;
class AssumptionCache;
} // namespace llvm

namespace hipsycl::compiler {
static constexpr size_t NumArrayElements = 1024;
struct MDKind {
  static constexpr const char Arrayified[] = "hipSYCL.arrayified";
  static constexpr const char InnerLoop[] = "hipSYCL.loop.inner";
  static constexpr const char WorkItemLoop[] = "hipSYCL.loop.workitem";
};

static constexpr const char BarrierIntrinsicName[] = "__hipsycl_barrier";

class SplitterAnnotationInfo;

namespace utils {
llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F);

bool isBarrier(const llvm::Instruction *I, const SplitterAnnotationInfo &SAA);
bool blockHasBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasBarriers(const llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool hasOnlyBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool startsWithBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
bool endsWithBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA);
llvm::CallInst *createBarrier(llvm::Instruction *InsertBefore, hipsycl::compiler::SplitterAnnotationInfo &SAA);

bool isWorkItemLoop(const llvm::Loop &L);
bool isInWorkItemLoop(const llvm::Loop &L);
bool isInWorkItemLoop(const llvm::Region &R, const llvm::LoopInfo &LI);
/*!
 * Get's the original work item loop.
 * @param LI The LoopInfo used to find the loop.
 * @return The single work item loop annotated with hipSYCL.loop.workitem.
 */
llvm::Loop *getSingleWorkItemLoop(const llvm::LoopInfo &LI);
llvm::BasicBlock *getWorkItemLoopBodyEntry(const llvm::Loop *WILoop);

bool checkedInlineFunction(llvm::CallBase *CI);

bool isAnnotatedParallel(llvm::Loop *TheLoop);

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup);

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup);

llvm::SmallPtrSet<llvm::BasicBlock *, 8> getBasicBlocksInWorkItemLoops(const llvm::LoopInfo &LI);

llvm::BasicBlock *splitEdge(llvm::BasicBlock *Root, llvm::BasicBlock *&Target, llvm::LoopInfo *LI,
                            llvm::DominatorTree *DT);
void promoteAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT, llvm::AssumptionCache &AC);
llvm::Instruction *getBrCmp(const llvm::BasicBlock &BB);

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
} // namespace utils
} // namespace hipsycl::compiler
#endif // HIPSYCL_IRUTILS_HPP
