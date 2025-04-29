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
#ifndef HIPSYCL_SYNC_ELISION_PASS_HPP
#define HIPSYCL_SYNC_ELISION_PASS_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <vector>

namespace hipsycl {
namespace compiler {

/// Implements the main synchronization elision logic. The idea is as follows:
/// 1.) Detect stdpar functions using "hipsycl_stdpar_entrypoint" annotation attribute.
/// 2.) Detect calls to __acpp_stdpar_optional_barrier() inside stdpar functions.
/// 3.) Remove __acpp_stdpar_optional_barrier() calls, and reinsert them after the call
/// instruction to the stdpar function (i.e. move them out of the stdpar function, and to the
/// callsite) 4.) Move calls to __acpp_stdpar_optional_barrier() down the instruction flow,
/// taking all routes through the control flow graph until a place is encountered where barriers
/// must be present for correctness:
///  - memory accesses such as loads/stores
///  - calls to other functions that are not stdpar calls, since we cannot know what these functions
///    do and our control flow analysis currently cannot continue in other functions
///  - exit of control flow from the current function
///
/// If a barrier is already present at one of the determined insertion points, no additional
/// barrier is inserted.
/// If a position where synchronization is needed is found, or a present barrier, that path in the
/// control flow graph is considered finished, and the other paths are investigated.
///
/// This algorithm effectively causes synchronization to be as delayed as possible, potentially
/// even removing synchronization entirely between two stdpar calls.
///
/// In order to properly function, __acpp_stdpar_optional_barrier() should have an internal
/// counter of enqueued operations, and only synchronize if that counter is > 0. This is because
/// in some control flow graphs it can happen that there is a path that crosses multiple
/// synchronization points.
///
/// Additional logic is applied when hitting store instructions: These are commonly inserted
/// by clang before stdpar calls to assemble function arguments, such as lambda objects passed
/// to the stdpar algorithm. This is problematic, because hitting these stores can prevent barriers
/// from moving beyond them, and ultimately prevent this optimization from becoming relevant in
/// most cases. Thus, a way is needed to distinguish stores that purely relate to assembling stdpar
/// arguments from stores to memory locations that might actually be used inside kernels.
///
/// In theory, we know that stores to addresses that are then exclusively used for by-value
/// arguments to stdpar calls can be considered to be part of stdpar argument handling, and can thus
/// be skipped. In practice, we cannot easily determine this due to opaque pointers and clang/LLVM's
/// inconsistent use of the byval(T) attribute. Currently, we use the following heuristic:
/// - Consider instructions stores to memory locations originate from an alloca
/// - The original alloca memory location and all its derived uses must only be in getelementptr
///   instructions, stores, and the stdpar call.
/// - The store must occur in the same block as the stdpar call, and precede it in the instruction
///   order.
///
/// It is possible that pathological cases can be constructed where this returns false positives,
/// especially in the presence of system USM where stack memory might be used inside kernels too.
/// In practice, for cases where this becomes relevant we should not offload anyway because the problem
/// size would be way too small to be an efficient offload use case.
class SyncElisionPass : public llvm::PassInfoMixin<SyncElisionPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

/// This pass causes callers of stdpar algorithms to be inlined. This is a simplistic heuristic
/// to combine more stdpar calls in one function, assuming that often stdpar usage happens from only
/// a few root functions. Having as many of the stdpar calls as possible in one function is important
/// because the main SyncElision algorithm currently does not work beyond a single function.
class SyncElisionInliningPass : public llvm::PassInfoMixin<SyncElisionInliningPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};


}
}

#endif