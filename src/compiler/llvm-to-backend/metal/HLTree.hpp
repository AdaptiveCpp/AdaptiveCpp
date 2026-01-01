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
#ifndef _HL_TREE_HPP_
#define _HL_TREE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// High-Level IR tree structures

namespace llvm {

class BasicBlock;
class Value;
class Function;
class raw_ostream;
class Loop;
class Region;

} // namespace llvm

namespace hipsycl {
namespace compiler {

namespace hl {

enum class NodeKind : uint8_t {
  List,
  Block,
  If,
  Loop,
  Break,
  Continue,
  Return,
};

struct Node {
  NodeKind kind;

  llvm::Region* R = nullptr;
  llvm::Loop* parentLoop = nullptr;

  Node() = default;
  explicit Node(NodeKind k) : kind(k) {}
  virtual ~Node() = default;

  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&&) = default;
  Node& operator=(Node&&) = default;
};

using NodePtr = std::shared_ptr<Node>;

struct ListNode final : Node {
  std::vector<NodePtr> items;

  ListNode() : Node(NodeKind::List) {}

  void append(NodePtr n) { items.push_back(std::move(n)); }
};

struct BlockNode final : Node {
  const llvm::BasicBlock* bb = nullptr;

  explicit BlockNode(const llvm::BasicBlock* b)
    : Node(NodeKind::Block), bb(b)
  {
  }
};

struct IfNode final : Node {
  // condition from `br i1 %cond, ...`
  const llvm::Value* cond = nullptr;

  NodePtr then_branch;
  NodePtr else_branch; // nullptr => no else

  IfNode(const llvm::Value* c, NodePtr thenBr, NodePtr elseBr = nullptr)
    : Node(NodeKind::If)
    , cond(c)
    , then_branch(std::move(thenBr))
    , else_branch(std::move(elseBr))
  {
  }
};

struct LoopNode final : Node {
  NodePtr body;

  LoopNode() = default;

  LoopNode(NodePtr body_)
    : Node(NodeKind::Loop)
    , body(std::move(body_))
  {
  }
};

struct BreakNode final : Node {
  BreakNode() : Node(NodeKind::Break)
  {}
};

struct ContinueNode final : Node {
  ContinueNode() : Node(NodeKind::Continue)
  {}
};

struct ReturnNode final : Node {
  const llvm::Value* value = nullptr; // nullptr => ret void

  explicit ReturnNode(const llvm::Value* v)
    : Node(NodeKind::Return)
    , value(v)
  {}
};

inline bool is(const Node& n, NodeKind k) {
  return n.kind == k;
}

template <class T>
inline T* as(Node* n) {
  return (n && n->kind == T{}.kind) ? static_cast<T*>(n) : nullptr;
}

void dumpFunctionTree(llvm::raw_ostream& os, const llvm::Function& F, const Node& root);

} // namespace hl

} // namespace compiler
} // namespace hipsycl

#endif