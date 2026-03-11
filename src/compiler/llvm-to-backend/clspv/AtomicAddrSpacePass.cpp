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
#include "hipSYCL/compiler/llvm-to-backend/clspv/AtomicAddrSpacePass.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

static llvm::SmallVector<llvm::StringRef> AtomicFuncs = {
    "_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope",
    "_Z20atomic_load_explicitPU3AS4VU7_Atomicl12memory_order12memory_scope",
    "_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z21atomic_store_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",
    "_Z24atomic_exchange_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z24atomic_exchange_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",

    "_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",

    "_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
    "scope",
    "_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
    "scope",

    "_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_"
    "AtomiciPU3AS4ii12memory_orderS4_12memory_scope",
    "_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_"
    "AtomiciPU3AS4jj12memory_orderS4_12memory_scope",

    "_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_"
    "AtomiciPU3AS4ii12memory_orderS4_12memory_scope",
    "_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_"
    "AtomiciPU3AS4jj12memory_orderS4_12memory_scope",
};

// Pass for fixing up atomic calls that take an opaque ptr without an address
// space but have an addrspace
llvm::PreservedAnalyses
AtomicAddrSpacePass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  bool DidTransform = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        auto Func = CI->getCalledFunction();
        auto Name = Func->getName();
        if (std::find(AtomicFuncs.begin(), AtomicFuncs.end(), Name) !=
            AtomicFuncs.end()) {
          auto Arg0 = CI->getArgOperand(0);
          auto ArgType = Arg0->getType();
          if (ArgType->getPointerAddressSpace() != 0) {
            llvm::IRBuilder Builder(CI);
            llvm::PointerType *GenType =
                llvm::PointerType::get(F.getParent()->getContext(), 0);
            auto AddrSpaceCast = Builder.CreateAddrSpaceCast(Arg0, GenType);
            CI->setArgOperand(0, AddrSpaceCast);
            DidTransform = true;
          }
        }
      }
    }
  }
  return !DidTransform ? llvm::PreservedAnalyses::all()
                       : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
