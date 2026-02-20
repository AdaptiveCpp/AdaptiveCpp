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
#ifndef EMITTER_HPP
#define EMITTER_HPP

#include <string>
#include <unordered_set>
#include <sstream>
#include <map>

#include "HLTree.hpp"

namespace llvm {
class Module;
class Function;
class Value;
class BasicBlock;
class Type;
} // namespace llvm

namespace hipsycl {
namespace compiler {

struct MetalEmitterOptions {
  // Metal supports at most 31 [[buffer(N)]] arguments in flat mode.
  // When a kernel has more than maxArgsForFlatMode parameters, all arguments
  // are packed into a single argument buffer struct instead.
  // Controlled at runtime via kernel_build_option::metal_max_args_for_flat_mode.
  int maxArgsForFlatMode = 6;
  std::unordered_map<unsigned, std::string> addressSpaceMap = {
    {0, "device"},
    {1, "device"},
    {3, "threadgroup"},
    {4, "constant"},
    {5, "thread"},
  };
};

class MetalEmitter {
public:
  MetalEmitter(llvm::Module& M, const std::unordered_set<std::string>& kernelNames, const MetalEmitterOptions& opt = {});
  bool emit(std::string& out);
  std::optional<std::string> errorMessage() const {
    return errorMsg;
  }

private:
  bool emitFunction(llvm::Function& F, const hl::Node& node);
  void emitTypes();
  void emitIntrinsicHelpers();
  bool emitArgStruct(llvm::Function& F);
  bool emitSignature(llvm::Function& F);
  bool emitDeclarations();
  bool emitNode(const hl::Node& node, int level);
  bool emitBasicBlock(const llvm::BasicBlock* BB, int level);
  bool emitInstruction(const llvm::Instruction& I, int level);
  void emitUnaryOperator(const llvm::UnaryOperator* UO, const std::string& name, int level);
  void emitBinaryOperator(const llvm::BinaryOperator* BO, const std::string& name, int level);
  void emitICmpInstruction(const llvm::ICmpInst* IC, const std::string& name, int level);
  void emitFCmpInstruction(const llvm::FCmpInst* FC, const std::string& name, int level);
  void emitGEPInstruction(const llvm::GetElementPtrInst* GEP, const std::string& name, int level);
  bool emitCallInstruction(const llvm::CallInst* CI, const std::string& name, int level);
  std::string emitExpr(const llvm::Value* V);
  std::string valueName(const llvm::Value* V);
  std::string basicBlockName(const llvm::BasicBlock* BB);
  std::string getSignedType(llvm::Type* T);
  std::string mapType(const llvm::Type* T);
  std::string mapType(const llvm::Value* V);
  std::string getAddressSpaceKeyword(unsigned AS);
  void analyzeCallInsts();
  void collectVariablesInfo(const llvm::Function& F);
  unsigned getPhysicalPointerAddressSpace(const llvm::Value* V);
  const llvm::Value* stripToRootObject(const llvm::Value* V);
  std::unordered_map<llvm::Function*, std::vector<llvm::Function*>> buildCallGraph();
  std::vector<llvm::Function*> topologicalSort(const std::unordered_map<llvm::Function*, std::vector<llvm::Function*>>& callGraph);
  bool emitMetalInlineCall(const llvm::CallInst* CI, const std::string& name, int level);

  MetalEmitterOptions opt;

  llvm::Module& M;
  std::unordered_set<std::string> kernelNames;

  // types and per-module info
  std::unordered_map<unsigned, std::string> addressSpaceMap;
  std::unordered_map<const llvm::Type*, std::string> typeCache;
  std::unordered_map<const llvm::Type*, std::string> anonStructs;
  int nextAnonymousStructId = 0;

  // local variables info
  std::unordered_map<const llvm::Instruction*, int> allocaIndex;
  std::unordered_map<const llvm::PHINode*, std::vector<const llvm::Value*>> phiSources;
  std::unordered_map<const llvm::Value*, std::vector<const llvm::PHINode*>> sourceToPhis;
  std::map<std::pair<const llvm::BasicBlock*, const llvm::PHINode*>, const llvm::Value*> phiIncomingFromBlock;
  std::unordered_set<const llvm::PHINode*> phiNodes;
  std::unordered_map<const llvm::Value*, std::string> valuesToDeclare;
  //
  std::unordered_map<const llvm::Value*, unsigned> inferredPtrAS;

  std::unordered_set<const llvm::Function*> needsDynamicLocalMemory;
  int inputStructCounter = 0;
  std::string inputStructName;

  std::ostringstream os;
  std::optional<std::string> errorMsg;
};

} // namespace compiler
} // namespace hipsycl

#endif // EMITTER_HPP