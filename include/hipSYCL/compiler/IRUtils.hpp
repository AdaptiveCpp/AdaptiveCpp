//
// Created by joachim on 04.06.21.
//

#ifndef HIPSYCL_IRUTILS_HPP
#define HIPSYCL_IRUTILS_HPP

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/LoopInfo.h>

namespace hipsycl::compiler::utils {
llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B,
                                 llvm::Function &F);

bool checkedInlineFunction(llvm::CallBase *CI);
} // namespace hipsycl::compiler::utils
#endif // HIPSYCL_IRUTILS_HPP
