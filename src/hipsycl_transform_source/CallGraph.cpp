//===- CallGraph.cpp - AST-based Call graph -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the AST-based CallGraph.
//
//===----------------------------------------------------------------------===//

/// This file is based on clang's CallGraph.cpp - there are only small changes
/// that let the CallGraph treat lambda functions as independent function declarations
/// with their own nodes in the call graph.

#include "CallGraph.hpp"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>

using namespace clang;
using namespace hipsycl::transform;

#define DEBUG_TYPE "CallGraph"

STATISTIC(NumObjCCallEdges, "Number of Objective-C method call edges");
STATISTIC(NumBlockCallEdges, "Number of block call edges");

namespace {

/// A helper class, which walks the AST and locates all the call sites in the
/// given function body.
class CGBuilder : public StmtVisitor<CGBuilder> {
  CallGraph *G;
  CallGraphNode *CallerNode;

public:
  CGBuilder(CallGraph *g, CallGraphNode *N) : G(g), CallerNode(N) {}

  void VisitStmt(Stmt *S) {
    // Only visit children if the statement is not a lambda expression -
    // otherwise we would falsely consider all functions called in the lambda
    // to be called from the function in which the lambda was declared.
    if(!isa<LambdaExpr>(S))
      VisitChildren(S);
    else
    {
      // If we have found a lambda, we must treat is as a new function and tell
      // the call graph about it.
      LambdaExpr* lambda = cast<LambdaExpr>(S);
      FunctionDecl* callOp = lambda->getCallOperator();
      if(callOp)
        G->VisitFunctionDecl(callOp);
    }
  }

  Decl *getDeclFromCall(CallExpr *CE) {
    if (FunctionDecl *CalleeDecl = CE->getDirectCallee())
      return CalleeDecl;

    // Simple detection of a call through a block.
    Expr *CEE = CE->getCallee()->IgnoreParenImpCasts();
    if (BlockExpr *Block = dyn_cast<BlockExpr>(CEE)) {
      NumBlockCallEdges++;
      return Block->getBlockDecl();
    }

    return nullptr;
  }

  void addCalledDecl(Decl *D) {
    // Don't ask CallGraph::includeInGraph() whether we should include that function
    // in the call graph since that would skip all templates even if they are instantiated
    // here

    CallGraphNode *CalleeNode = G->getOrInsertNode(D);
    CallerNode->addCallee(CalleeNode);
  }

  void VisitCallExpr(CallExpr *CE) {
    if (Decl *D = getDeclFromCall(CE))
      addCalledDecl(D);
    VisitChildren(CE);
  }

  // Adds may-call edges for the ObjC message sends.
  void VisitObjCMessageExpr(ObjCMessageExpr *ME) {
    if (ObjCInterfaceDecl *IDecl = ME->getReceiverInterface()) {
      Selector Sel = ME->getSelector();

      // Find the callee definition within the same translation unit.
      Decl *D = nullptr;
      if (ME->isInstanceMessage())
        D = IDecl->lookupPrivateMethod(Sel);
      else
        D = IDecl->lookupPrivateClassMethod(Sel);
      if (D) {
        addCalledDecl(D);
        NumObjCCallEdges++;
      }
    }
  }

  void VisitChildren(Stmt *S) {
    for (Stmt *SubStmt : S->children())
      if (SubStmt)
        this->Visit(SubStmt);
  }
};

} // namespace

namespace hipsycl {
namespace transform {

void CallGraph::addNodesForBlocks(DeclContext *D) {
  if (BlockDecl *BD = dyn_cast<BlockDecl>(D))
    addNodeForDecl(BD, true);

  for (auto *I : D->decls())
    if (auto *DC = dyn_cast<DeclContext>(I))
      addNodesForBlocks(DC);
}

CallGraph::CallGraph() {
  Root = getOrInsertNode(nullptr);
}

CallGraph::~CallGraph() = default;

bool CallGraph::includeInGraph(const Decl *D) {
  assert(D);
  if (!D->hasBody())
    return false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // We skip function template definitions, as their semantics is
    // only determined when they are instantiated.
    //if (FD->isDependentContext())
    //  return false;

    IdentifierInfo *II = FD->getIdentifier();
    if (II && II->getName().startswith("__inline"))
      return false;
  }

  return true;
}

void CallGraph::addNodeForDecl(Decl* D, bool IsGlobal) {
  assert(D);

  // Allocate a new node, mark it as root, and process it's calls.
  CallGraphNode *Node = getOrInsertNode(D);

  // Process all the calls by this function as well.
  CGBuilder builder(this, Node);
  if (Stmt *Body = D->getBody())
    builder.Visit(Body);
}

CallGraphNode *CallGraph::getNode(const Decl *F) const {
  FunctionMapTy::const_iterator I = FunctionMap.find(F);
  if (I == FunctionMap.end()) return nullptr;
  return I->second.get();
}

CallGraphNode *CallGraph::getOrInsertNode(Decl *F) {
  if (F && !isa<ObjCMethodDecl>(F))
    F = F->getCanonicalDecl();

  std::unique_ptr<CallGraphNode> &Node = FunctionMap[F];
  if (Node)
    return Node.get();

  Node = llvm::make_unique<CallGraphNode>(F);
  // Make Root node a parent of all functions to make sure all are reachable.
  if (F)
    Root->addCallee(Node.get());
  return Node.get();
}

void CallGraphNode::print(raw_ostream &os) const {
  if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(FD))
      return ND->printName(os);
  os << "< >";
}


}
}

