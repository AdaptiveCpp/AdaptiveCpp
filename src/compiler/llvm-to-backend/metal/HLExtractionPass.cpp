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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include "HLExtractionPass.hpp"

#include <deque>
#include <iostream>

#define DEBUG_VERIFY_COMPLETENESS

/*
Join-slicing HL extraction
=======================================================

Prerequisites: The input LLVM IR has already been run through CFG-structuring passes
Available analyses: LoopInfo, RegionInfo, PostDominatorTree (PDT)

Two-pass pipeline:
-----------------
Pass 1 (extractLoops): Build region/loop skeleton from RegionInfo and LoopInfo
Pass 2 (extractConditions): Rebuild CFG, structurize if-then-else via PDT join-slicing

Key algorithm: For conditional branch C->{T,E}, find J=PDT(T,E), emit:
    C; if(cond) {T..J} else {E..J}; continue at J

CORRECTNESS PRINCIPLE: Never skip blocks. If structuring fails, duplicate code.

Handled patterns:
-----------------

1) IDEAL: Disjoint diamond
         [C]              Both branches disjoint -> clean if-then-else
        /   \
     [T]     [E]
        \   /
         [J]

2) OVERLAPPING: Shared blocks (rare, only in irreducible CFG)
       [C]                BB4 reachable from both T and E before join
      /   \               -> Duplicate BB4 in both branches using allowedSet
   [T]   [E]
      \ / \
     [BB4] |
        \ /
        [J]

3) FALLBACK 1: Join outside region/loop
   Region R:
     [C]                  Join exists but outside boundaries
    /   \                 -> Emit both branches fully up to region/loop limit
  [T]   [E]
   |     |
   exit R
     |
    [J] <- outside R

4) FALLBACK 2: No join (divergent paths)
     [C]                  PDT returns nullptr (different returns/exceptions)
    /   \                 -> Emit both branches fully
  [T]   [E]
   |     |
  ret1  ret2
*/

namespace hipsycl {
namespace compiler {
namespace hl {

using namespace llvm;

namespace {

struct Extractor {
  Extractor(const llvm::Function& F,
            const llvm::LoopInfo& LI,
            const llvm::PostDominatorTree& PDT,
            const llvm::RegionInfo& RI)
    : F_(F), LI_(LI), PDT_(PDT), RI_(RI)
  {}

  NodePtr extractFunction() {
    auto tree = extractLoops();       // stage1: regions + loops
    extractConditions(tree);          // stage2: if-then-else structuring
    return tree;
  }

  NodePtr extractLoops() {
    auto* R = RI_.getTopLevelRegion();
    return EmitRegion(R, nullptr, 1);
  }

  void extractConditions(NodePtr n) {
    switch (n->kind) {
      case NodeKind::List: {
        auto ln = std::static_pointer_cast<ListNode>(n);
        if (ln->R) {
          // Region list: rebuild via CFG traversal + PDT slicing
          rebuildRegionListInPlace(*ln);
        } else {
          // Non-region list: recursively process children
          auto oldItems = std::move(ln->items);
          std::vector<NodePtr> items;
          items.reserve(oldItems.size());
          for (auto&& item : oldItems) {
            if (item && item->kind == NodeKind::Block) {
              item = EmitBlock(
                std::static_pointer_cast<BlockNode>(item)->bb,
                n->R,
                n->parentLoop);
            } else {
                extractConditions(item);
            }
            items.emplace_back(std::move(item));
          }
          ln->items = std::move(items);
        }
        break;
      }
      case NodeKind::Loop: {
        auto ln = std::static_pointer_cast<LoopNode>(n);
        if (ln->body) {
          extractConditions(ln->body);
        }
        break;
      }
      default:
        break;
    }
  }

  // ---------------------- stage2 helpers ----------------------

  struct EntryNodeIndex {
    std::unordered_map<const BasicBlock*, NodePtr> entryToNode;
  };

  EntryNodeIndex indexDirectChildrenByEntry(const ListNode& ln) {
    EntryNodeIndex idx;
    for (const auto& item : ln.items) {
      if (!item) continue;
      if (item->kind == NodeKind::Block) continue;
      if (!item->R) continue;
      auto* entry = item->R->getEntry();
      if (!entry) continue;
      idx.entryToNode.emplace(entry, item);
    }
    return idx;
  }

  void markRegionBlocksConsumed(
    Region* R,
    std::unordered_set<const BasicBlock*>& consumed) const
  {
    if (!R) return;
    auto* Entry = R->getEntry();
    auto* Exit = R->getExit();
    if (!Entry) return;

    std::deque<const BasicBlock*> q;
    q.push_back(Entry);
    while (!q.empty()) {
      auto* BB = q.front();
      q.pop_front();
      if (!BB) continue;
      if (!R->contains(BB)) continue;
      if (BB == Exit) continue;
      if (!consumed.emplace(BB).second) continue;
      for (const auto* S : successors(BB)) {
        if (S && R->contains(S) && S != Exit) {
          q.push_back(S);
        }
      }
    }
  }

  // BFS from start, collecting all reachable blocks until stop (exclusive)
  std::unordered_set<const BasicBlock*> collectBranchBlocksUntil(
    const BasicBlock* start,
    const BasicBlock* stop,
    Region* regionLimit,
    Loop* loopLimit) const
  {
    std::unordered_set<const BasicBlock*> out;
    if (!start || start == stop) return out;

    std::deque<const BasicBlock*> q;
    q.push_back(start);
    while (!q.empty()) {
      auto* BB = q.front();
      q.pop_front();
      if (!BB || BB == stop) continue;
      if (regionLimit && !regionLimit->contains(BB)) continue;
      if (loopLimit && !loopLimit->contains(BB)) continue;
      if (!out.emplace(BB).second) continue;
      for (const auto* S : successors(BB)) {
        if (!S || S == stop) continue;
        if (regionLimit && !regionLimit->contains(S)) continue;
        if (loopLimit && !loopLimit->contains(S)) continue;
        q.push_back(S);
      }
    }
    return out;
  }

  const BasicBlock* computeJoinForBranch(
    const BasicBlock* thenBB,
    const BasicBlock* elseBB) const
  {
    if (!thenBB || !elseBB) return nullptr;
    return PDT_.findNearestCommonDominator(thenBB, elseBB);
  }

  NodePtr buildSequence(
    Region* R,
    Loop* parentLoop,
    const BasicBlock* start,
    const BasicBlock* stop,
    const EntryNodeIndex& idx,
    std::unordered_set<const BasicBlock*>& globalConsumed,
    const std::unordered_set<const BasicBlock*>* allowedSet = nullptr)
  {
    auto seq = std::make_shared<ListNode>();
    seq->R = R;
    seq->parentLoop = parentLoop;

    const BasicBlock* BB = start;
    std::unordered_set<const BasicBlock*> localConsumed;

    auto isAllowed = [&](const BasicBlock* X) -> bool {
      if (!X) return false;
      if (X == stop) return false;
      if (R && !R->contains(X)) return false;
      if (parentLoop && !parentLoop->contains(X)) return false;
      if (allowedSet && allowedSet->find(X) == allowedSet->end()) return false;
      return true;
    };

    while (BB && BB != stop) {
      if (!isAllowed(BB)) break;
      if (localConsumed.find(BB) != localConsumed.end()) break;

      // Child region: emit as-is, skip to its exit
      if (auto it = idx.entryToNode.find(BB); it != idx.entryToNode.end()) {
        NodePtr child = it->second;
        if (child) {
          extractConditions(child);
          seq->append(child);
          markRegionBlocksConsumed(child->R, globalConsumed);
          BB = child->R ? child->R->getExit() : nullptr;
          continue;
        }
      }

      localConsumed.emplace(BB);
      globalConsumed.emplace(BB);

      auto bn = std::make_shared<BlockNode>(BB);
      bn->R = R;
      bn->parentLoop = parentLoop;
      seq->append(bn);

      const Instruction* term = BB->getTerminator();
      if (!term) break;

      if (auto* ret = dyn_cast<ReturnInst>(term)) {
        seq->append(std::make_shared<ReturnNode>(ret->getReturnValue()));
        break;
      }

      auto* br = dyn_cast<BranchInst>(term);
      if (!br) break;

      if (br->isUnconditional()) {
        auto* succ = br->getSuccessor(0);
        if (!succ) break;
        if (parentLoop) {
          if (succ == parentLoop->getHeader()) {
            seq->append(std::make_shared<ContinueNode>());
            break;
          }
          if (!parentLoop->contains(succ) || succ == parentLoop->getExitBlock()) {
            seq->append(std::make_shared<BreakNode>());
            break;
          }
        }
        if (R && !R->contains(succ)) break;
        BB = succ;
        continue;
      }

      // ===== CONDITIONAL BRANCH: if-then-else structuring =====

      auto* thenBB = br->getSuccessor(0);
      auto* elseBB = br->getSuccessor(1);
      if (!thenBB || !elseBB) break;

      // Special case: loop continue/break
      if (parentLoop) {
        const auto* header = parentLoop->getHeader();
        bool thenIsHeader = (thenBB == header);
        bool elseIsHeader = (elseBB == header);
        bool thenInLoop = parentLoop->contains(thenBB);
        bool elseInLoop = parentLoop->contains(elseBB);

        if (thenIsHeader && !elseInLoop) {
          seq->append(std::make_shared<IfNode>(
          br->getCondition(),
          std::make_shared<ContinueNode>(),
          std::make_shared<BreakNode>()));
          break;
        }
        if (elseIsHeader && !thenInLoop) {
          seq->append(std::make_shared<IfNode>(
          br->getCondition(),
          std::make_shared<BreakNode>(),
          std::make_shared<ContinueNode>()));
          break;
        }
      }

      const BasicBlock* join = computeJoinForBranch(thenBB, elseBB);

      bool joinOk = join && join != BB;
      if (joinOk) {
        // Join can be region exit (OK), but not outside region
        if (R && join != R->getExit() && !R->contains(join)) {
          joinOk = false;
        }
        if (parentLoop && join && join != parentLoop->getExitBlock() && !parentLoop->contains(join)) {
          joinOk = false;
        }
      }

      // Unified if-node emitter for all cases
      // useSharedConsumed: true for disjoint (efficient), false for overlapping/fallback (safe)
      auto emitIfNode = [&](
        const BasicBlock* joinPoint,
        const std::unordered_set<const BasicBlock*>* thenAllowedSet,
        const std::unordered_set<const BasicBlock*>* elseAllowedSet,
        bool useSharedConsumed) -> NodePtr
      {
        NodePtr thenN, elseN;

        if (useSharedConsumed) {
          // IDEAL: both branches share globalConsumed (no duplication)
          thenN = (thenAllowedSet && thenAllowedSet->empty()) ? nullptr
            : buildSequence(R, parentLoop, thenBB, joinPoint, idx, globalConsumed, thenAllowedSet);
          elseN = (elseAllowedSet && elseAllowedSet->empty()) ? nullptr
            : buildSequence(R, parentLoop, elseBB, joinPoint, idx, globalConsumed, elseAllowedSet);
        } else {
          // OVERLAPPING/FALLBACK: separate consumed sets, merge after
          std::unordered_set<const BasicBlock*> thenConsumed = globalConsumed;
          std::unordered_set<const BasicBlock*> elseConsumed = globalConsumed;

          thenN = (thenAllowedSet && thenAllowedSet->empty()) ? nullptr
            : buildSequence(R, parentLoop, thenBB, joinPoint, idx, thenConsumed, thenAllowedSet);
          elseN = (elseAllowedSet && elseAllowedSet->empty()) ? nullptr
            : buildSequence(R, parentLoop, elseBB, joinPoint, idx, elseConsumed, elseAllowedSet);

          globalConsumed.insert(thenConsumed.begin(), thenConsumed.end());
          globalConsumed.insert(elseConsumed.begin(), elseConsumed.end());
        }

        auto ifN = std::make_shared<IfNode>(br->getCondition(), std::move(thenN), std::move(elseN));
        ifN->R = R;
        ifN->parentLoop = parentLoop;
        return ifN;
      };

      if (joinOk) {
        auto thenSet = collectBranchBlocksUntil(thenBB, join, R, parentLoop);
        auto elseSet = collectBranchBlocksUntil(elseBB, join, R, parentLoop);

        // Check if branches are disjoint (don't share blocks)
        bool disjoint = true;
        for (const auto* X : thenSet) {
          if (elseSet.find(X) != elseSet.end()) { disjoint = false; break; }
        }
        if (disjoint) {
          for (const auto* X : thenSet) {
            if (globalConsumed.find(X) != globalConsumed.end()) { disjoint = false; break; }
          }
        }
        if (disjoint) {
          for (const auto* X : elseSet) {
            if (globalConsumed.find(X) != globalConsumed.end()) { disjoint = false; break; }
          }
        }

        if (disjoint) {
          // IDEAL: clean diamond, no duplication
          seq->append(emitIfNode(join, &thenSet, &elseSet, true));
          BB = join;
          continue;
        }

        // OVERLAPPING: branches share blocks before join
        //    [C]              Happens in irreducible CFG
        //   /   \             allowedSet constrains each branch
        // [T]   [E]           to prevent wandering
        //   \ / \            -> Duplicate shared blocks
        //   [S] |
        //    \ /
        //    [J]
        {
          seq->append(emitIfNode(join, &thenSet, &elseSet, false));
          BB = join;
          continue;
        }
      }

      // FALLBACK 1: join outside region/loop
      // Region R:
      //   [C]              Join found but outside scope
      //  /   \             -> Emit both branches to boundary
      // [T]  [E]
      //  |    |
      //  exit R
      //   |
      //  [J] <- outside
      if (join) {
        seq->append(emitIfNode(stop, nullptr, nullptr, false));
        break;
      }

      // FALLBACK 2: no join (divergent control flow)
      //   [C]              PDT returns nullptr
      //  /   \             (different returns, exceptions, infinite loop)
      // [T]  [E]           -> Emit both branches fully
      //  |    |
      // ret  ret
      {
        seq->append(emitIfNode(stop, nullptr, nullptr, false));
      }
      break;
    }

    return seq;
  }

  void rebuildRegionListInPlace(ListNode& ln) {
    Region* R = ln.R;
    if (!R) return;

    EntryNodeIndex idx = indexDirectChildrenByEntry(ln);
    std::unordered_set<const BasicBlock*> consumed;

    NodePtr rebuilt = buildSequence(R, ln.parentLoop, R->getEntry(), nullptr, idx, consumed);
    auto rebuiltList = std::static_pointer_cast<ListNode>(rebuilt);
    ln.items = std::move(rebuiltList->items);

#ifdef DEBUG_VERIFY_COMPLETENESS
    auto* Entry = R->getEntry();
    auto* Exit = R->getExit();
    std::unordered_set<const BasicBlock*> regionBlocks;
    std::deque<const BasicBlock*> q;
    q.push_back(Entry);
    while (!q.empty()) {
      auto* BB = q.front();
      q.pop_front();
      if (!BB || !R->contains(BB) || BB == Exit) continue;
      if (!regionBlocks.insert(BB).second) continue;
      for (auto* S : successors(BB)) {
        if (S && R->contains(S) && S != Exit) {
          q.push_back(S);
        }
      }
    }

    for (auto* BB : regionBlocks) {
      if (consumed.find(BB) == consumed.end()) {
        std::cerr << "WARNING: Block not emitted in region: " << BB->getName().str() << "\n";
      }
    }
#endif
  }

  NodePtr EmitConditional(const BranchInst* br, const llvm::BasicBlock* BB, Region* R, Loop* parentLoop) {
    auto* then_ = br->getSuccessor(0);
    auto* else_ = br->getSuccessor(1);

    if (!parentLoop || (parentLoop->contains(then_) && parentLoop->contains(else_))) {
      return std::make_shared<IfNode>(
        br->getCondition(),
        EmitBlock(then_, R, parentLoop),
        EmitBlock(else_, R, parentLoop));
    }

    if (then_ == parentLoop->getHeader() && !parentLoop->contains(else_)) {
      return std::make_shared<IfNode>(
        br->getCondition(),
        std::make_shared<ContinueNode>(),
        std::make_shared<BreakNode>());
    }
    if (then_ == parentLoop->getHeader() && parentLoop->contains(else_)) {
      return std::make_shared<IfNode>(
        br->getCondition(),
        std::make_shared<ContinueNode>(),
        EmitBlock(else_, R, parentLoop));
    }
    if (else_ == parentLoop->getHeader() && !parentLoop->contains(then_)) {
      return std::make_shared<IfNode>(
        br->getCondition(),
        std::make_shared<BreakNode>(),
        std::make_shared<ContinueNode>());
    }
    if (else_ == parentLoop->getHeader() && parentLoop->contains(then_)) {
      return std::make_shared<IfNode>(
        br->getCondition(),
        EmitBlock(then_, R, parentLoop),
        std::make_shared<ContinueNode>());
    }

    return std::make_shared<IfNode>(
      br->getCondition(),
      then_ ? EmitBlock(then_, R, parentLoop) : nullptr,
      else_ ? EmitBlock(else_, R, parentLoop) : nullptr);
  }

  NodePtr EmitUnconditional(const BranchInst* br, const llvm::BasicBlock* BB, Region* R, Loop* parentLoop) {
    auto* succ = br->getSuccessor(0);
    if (!succ) return nullptr;

    if (parentLoop) {
      if (succ == parentLoop->getHeader()) {
        return std::make_shared<ContinueNode>();
      }
      if (succ == parentLoop->getExitBlock() || !parentLoop->contains(succ)) {
        return std::make_shared<BreakNode>();
      }
      return std::make_shared<BlockNode>(succ);
    }

    if (R && !R->contains(succ)) {
      return nullptr;
    }

    return std::make_shared<BlockNode>(succ);
  }

  NodePtr EmitBlock(const llvm::BasicBlock* BB, Region* R, Loop* parentLoop) {
    if (!BB) return nullptr;

    auto* term = BB->getTerminator();

    auto listNode = std::make_shared<ListNode>();
    listNode->R = R;
    listNode->parentLoop = parentLoop;
    listNode->append(std::make_shared<BlockNode>(BB));

    if (auto* ret = dyn_cast_or_null<ReturnInst>(term)) {
      listNode->append(std::make_shared<ReturnNode>(ret->getReturnValue()));
      return listNode;
    }

    if (auto* br = dyn_cast_or_null<BranchInst>(term)) {
      NodePtr next = nullptr;
      if (br->isUnconditional()) {
        next = EmitUnconditional(br, BB, R, parentLoop);
      } else {
        next = EmitConditional(br, BB, R, parentLoop);
      }
      if (next) {
        listNode->append(std::move(next));
      }
    }

    return listNode;
  }

  NodePtr EmitRegion(Region* R, Loop* parentLoop, int level) {
    if (!R) return nullptr;

    bool loopStart = false;
    auto* loop = LI_.getLoopFor(R->getEntry());
    if (loop != parentLoop && loop && loop->getHeader() == R->getEntry()) {
      loopStart = true;
    }

    NodePtr res;
    ListNode* listNode = nullptr;
    if (loopStart) {
      auto body = std::make_shared<ListNode>();
      listNode = body.get();
      res = std::make_shared<LoopNode>(std::move(body));
    } else {
      auto body = std::make_shared<ListNode>();
      listNode = body.get();
      res = std::move(body);
    }

    auto* Entry = R->getEntry();
    auto* Exit = R->getExit();

    if (!Entry) return res;

    SmallPtrSet<const BasicBlock*, 32> visited;
    std::deque<BasicBlock*> work;
    work.push_back(const_cast<BasicBlock*>(Entry));

    res->R = R;
    res->parentLoop = loop;

    listNode->R = R;
    listNode->parentLoop = loop;

    while (!work.empty()) {
      auto* BB = work.front();
      work.pop_front();

      if (!R->contains(BB)) continue;
      if (BB == Exit) continue;
      if (!visited.insert(BB).second) continue;

      auto* maybeChild = R->getSubRegionNode(BB);
      if (maybeChild && maybeChild != R) {
        auto* Child = maybeChild;
        listNode->append(EmitRegion(Child, loop, level + 1));
        auto* childExit = Child->getExit();
        if (childExit && R->contains(childExit)) {
          work.push_back(const_cast<BasicBlock*>(childExit));
        }
        continue;
      }

      auto blockNode = std::make_shared<BlockNode>(BB);
      blockNode->R = R;
      blockNode->parentLoop = loop;
      listNode->append(blockNode);

      auto* term = BB->getTerminator();
      if (!term) continue;

      for (auto* S : successors(BB)) {
        if (S && R->contains(S) && S != Exit) {
          work.push_back(const_cast<BasicBlock*>(S));
        }
      }
    }

    return res;
  }

  bool isLoopRegion(Region* R) const {
    if (!R) return false;
    Loop* L = LI_.getLoopFor(R->getEntry());
    return L && L->getHeader() == R->getEntry();
  }

private:
  const llvm::Function& F_;
  const llvm::LoopInfo& LI_;
  const llvm::PostDominatorTree& PDT_;
  const llvm::RegionInfo& RI_;
};

} // namespace

PreservedAnalyses HLExtractionPass::run(llvm::Function& F, llvm::FunctionAnalysisManager& FAM) {
  auto& LI  = FAM.getResult<llvm::LoopAnalysis>(F);
  auto& PDT = FAM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto& RI = FAM.getResult<llvm::RegionInfoAnalysis>(F);

  Extractor E(F, LI, PDT, RI);
  tree = E.extractFunction();

  return llvm::PreservedAnalyses::none();
}

} // namespace hl
} // namespace compiler
} // namespace hipsycl