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
#include "HLTree.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace hipsycl {
namespace compiler {
namespace hl {

namespace {

using namespace llvm;

void dumpNode(raw_ostream& os, const Node& n, int indentSpaces);

void indent(raw_ostream& os, int n) {
  for (int i = 0; i < n; ++i) {
    os << ' ';
  }
}

void printValueLikeIR(raw_ostream& os, const Value* v) {
  if (!v) {
    os << "<null>"; return;
  }
  std::string s;
  raw_string_ostream ss(s);
  v->print(ss);
  ss.flush();
  os << s;
}

void printInstrLikeIR(raw_ostream& os, const Instruction& I) {
  std::string s;
  raw_string_ostream ss(s);
  I.print(ss);
  ss.flush();
  os << s;
}

void printBlockPayloadLikeIR(
  raw_ostream& os,
  const BasicBlock* bb,
  int indentSpaces)
{
  if (!bb) {
    return;
  }

  const Instruction* TI = bb->getTerminator();
  for (const Instruction& I : *bb) {
    if (&I == TI) break;

    indent(os, indentSpaces);

    printInstrLikeIR(os, I);
    os << "\n";
  }
}

void printStructuredBranchOnCond(raw_ostream& os,
                                 const Value* cond,
                                 bool condTrueToContinue,
                                 int indentSpaces)
{
  indent(os, indentSpaces);
  os << "; loop guard\n";

  indent(os, indentSpaces);
  os << "if (";
  if (condTrueToContinue) {
    os << "!(";
    printValueLikeIR(os, cond);
    os << ")";
  } else {
    printValueLikeIR(os, cond);
  }
  os << ") {\n";

  indent(os, indentSpaces + 2);
  os << "break;\n";

  indent(os, indentSpaces);
  os << "}\n";
}

void dumpList(raw_ostream& os, const ListNode& ln, int indentSpaces) {
  for (const auto& item : ln.items) {
    if (!item) {
      continue;
    }
    dumpNode(os, *item, indentSpaces);
  }
}

void dumpBlock(raw_ostream& os, const BlockNode& bn, int indentSpaces) {
  indent(os, indentSpaces);
  if (bn.bb) {
    bn.bb->printAsOperand(os, /*PrintType=*/false);
  } else {
    os << "<null>";
  }
  os << ":\n";

  printBlockPayloadLikeIR(os, bn.bb, indentSpaces + 2);
}

void dumpIf(raw_ostream& os, const IfNode& in, int indentSpaces) {
  indent(os, indentSpaces);
  os << "if (";
  printValueLikeIR(os, in.cond);
  os << ") {\n";

  if (in.then_branch) dumpNode(os, *in.then_branch, indentSpaces + 2);

  indent(os, indentSpaces);
  os << "}";

  if (in.else_branch) {
    os << " else {\n";
    dumpNode(os, *in.else_branch, indentSpaces + 2);
    indent(os, indentSpaces);
    os << "}";
  }
  os << "\n";
}

void dumpLoop(raw_ostream& os, const LoopNode& ln, int indentSpaces) {
  indent(os, indentSpaces);
  os << "loop {\n";

  if (ln.body) {
    dumpNode(os, *ln.body, indentSpaces + 2);
  }

  indent(os, indentSpaces);
  os << "}\n";
}

void dumpReturn(raw_ostream& os, const ReturnNode& rn, int indentSpaces) {
  indent(os, indentSpaces);
  os << "ret";
  if (rn.value) {
    os << " ";
    printValueLikeIR(os, rn.value);
  }
  os << "\n";
}

void dumpNode(raw_ostream& os, const Node& n, int indentSpaces) {
  switch (n.kind) {
    case NodeKind::List:
      dumpList(os, static_cast<const ListNode&>(n), indentSpaces);
      break;
    case NodeKind::Block:
      dumpBlock(os, static_cast<const BlockNode&>(n), indentSpaces);
      break;
    case NodeKind::If:
      dumpIf(os, static_cast<const IfNode&>(n), indentSpaces);
      break;
    case NodeKind::Loop:
      dumpLoop(os, static_cast<const LoopNode&>(n), indentSpaces);
      break;
    case NodeKind::Break:
      indent(os, indentSpaces);
      os << "break\n";
      break;
    case NodeKind::Continue:
      indent(os, indentSpaces);
      os << "continue\n";
      break;
    case NodeKind::Return:
      dumpReturn(os, static_cast<const ReturnNode&>(n), indentSpaces);
      break;
    default:
      indent(os, indentSpaces);
      os << "<unknown-node>\n";
      break;
  }
}

} // namespace

void dumpFunctionTree(
  raw_ostream& os,
  const Function& F,
  const Node& root)
{
  os << "define @" << F.getName() << " {\n";
  dumpNode(os, root, /*indent=*/2);
  os << "}\n";
}

} // namespace hl

} // namespace compiler
} // namespace hipsycl