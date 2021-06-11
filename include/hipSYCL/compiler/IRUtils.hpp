//
// Created by joachim on 04.06.21.
//

#ifndef HIPSYCL_IRUTILS_HPP
#define HIPSYCL_IRUTILS_HPP

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/LoopInfo.h>

namespace hipsycl::compiler {
static constexpr size_t NumArrayElements = 1024;
struct MDKind {
  static constexpr const char Arrayified[] = "hipSYCL.arrayified";
  static constexpr const char InnerLoop[] = "hipSYCL.loop.inner";
  static constexpr const char WorkItemLoop[] = "hipSYCL.loop.workitem";
};

namespace utils {
llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F);

bool checkedInlineFunction(llvm::CallBase *CI);

bool isAnnotatedParallel(llvm::Loop *TheLoop);

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup);

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup);

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