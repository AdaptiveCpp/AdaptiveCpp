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
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>

#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Scalar/Reg2Mem.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ADCE.h"

#include "Emitter.hpp"
#include "HLTree.hpp"
#include "HLExtractionPass.hpp"

using namespace llvm;

namespace hipsycl {
namespace compiler {

using namespace hl;

namespace {

std::string indent(int level) {
  return std::string(level * 2, ' ');
}

std::string instToString(const llvm::Instruction& I) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  I.print(rso);
  return rso.str();
}

} // namespace

MetalEmitter::MetalEmitter(Module& M, const std::unordered_set<std::string>& kernelNames, const MetalEmitterOptions& opt)
  : M(M), kernelNames(kernelNames), opt(opt)
  , addressSpaceMap{opt.addressSpaceMap}
  , externalFunctionMapper(
    [this](const Value* V) -> std::string {
      unsigned AS = getPhysicalPointerAddressSpace(V);
      auto it = addressSpaceMap.find(AS);
      if (it != addressSpaceMap.end()) {
        return it->second;
      } else {
        return std::string("unknown_address_space");
      }
    },
    [this](const Value* V) -> std::string {
      return emitExpr(V);
    },
    [this](const Type* T) -> std::string {
      return mapType(T);
    }
  )
{
}

bool MetalEmitter::emit(std::string& out) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  analyzeCallInsts();

  os << R"__(
#include <metal_stdlib>
using namespace metal;

// These structs are used for void** dereferencing
struct __struct_ptr_to_device {
  device void* value;
};

struct __struct_ptr_to_threadgroup {
  threadgroup void* value;
};

struct __struct_ptr_to_constant {
  constant void* value;
};

struct __struct_ptr_to_thread {
  thread void* value;
};
)__";

  emitTypes();
  emitIntrinsicHelpers();

  for (Function& F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    HLExtractionPass hlPass;
    hlPass.run(F, FAM);
    if (!hlPass.tree) {
      errorMsg = "Failed to extract HL tree for " + F.getName().str();
      return false;
    }

    if (!emitFunction(F, *hlPass.tree)) {
      return false;
    }
  }

  if (errorMsg.has_value()) {
    return false;
  }

  out = os.str();
  return true;
}

bool MetalEmitter::emitFunction(Function& F, const Node& node) {
  collectVariablesInfo(F);

  bool success = emitArgStruct(F) &&
    emitSignature(F) &&
    emitDeclarations() &&
    emitNode(node, 1);

  if (!success) {
    return false;
  }

  os << "}\n\n";
  return true;
}

void MetalEmitter::emitTypes() {
  std::vector<const StructType*> structs;
  for (const StructType *ST : M.getIdentifiedStructTypes()) {
    structs.push_back(ST);
  }
  for (auto& [T, _] : anonStructs) {
    if (auto* ST = dyn_cast<StructType>(T)) {
      structs.push_back(ST);
    }
  }

  std::unordered_map<const StructType*, std::unordered_set<const StructType*>> deps;

  auto addDeps = [&](const StructType *ST) {
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *elemTy = ST->getElementType(i);
      if (auto *elemST = dyn_cast<StructType>(elemTy)) {
        deps[ST].insert(elemST);
      }
    }
  };

  for (const StructType *ST : structs) {
    addDeps(ST);
  }

  std::unordered_set<const StructType*> emitted;
  std::function<void(const StructType*)> emitType = [&](const StructType *ST) {
    if (!ST) return;
    if (emitted.count(ST)) return;

    for (const StructType *dep : deps[ST]) {
      emitType(dep);
    }

    emitted.insert(ST);

    os << "struct " << mapType(ST) << " { \n";
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      const Type *elemTy = ST->getElementType(i);
      auto typeName = mapType(elemTy);
      os << indent(1) << typeName << " field" << i << ";\n";
    }
    os << "};\n\n";
  };

  for (const StructType *ST : structs) {
    emitType(ST);
  }
}

void MetalEmitter::emitIntrinsicHelpers() {
  for (Function& F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (const BasicBlock& BB : F) {
      for (const Instruction& I : BB) {
        if (!I.getType()->isVoidTy()) {
          // mark used type-helpers
          mapType(I.getType());
        }
      }
    }
  }

  for (const auto& funcInfo : externalFunctionMapper.getUsedFunctions()) {
    if (funcInfo->code) {
      os << funcInfo->code.value() << "\n\n";
    }
  }
}

bool MetalEmitter::emitArgStruct(Function& F) {
  bool isKernel = kernelNames.count(F.getName().str()) > 0;
  if (!isKernel) {
    return true;
  }
  bool useArgStruct = isKernel && F.arg_size() > opt.maxArgsForFlatMode;
  if (!useArgStruct) {
    return true;
  }

  inputStructName = "Input_" + std::to_string(inputStructCounter++);
  os << "struct " << inputStructName << " {\n";
  int fieldIdx = 0;
  for (Argument& A : F.args()) {
    auto typeName = mapType(A.getType());
    os << indent(1) << typeName << " " << valueName(&A) << " [[id(" << fieldIdx++ << ")]];\n";
  }
  os << "};\n\n";
  return true;
}

bool MetalEmitter::emitSignature(Function& F) {
  bool isKernel = kernelNames.count(F.getName().str()) > 0;
  bool needDlm = needsDynamicLocalMemory.count(&F) > 0;

  bool useArgStruct = isKernel && F.arg_size() > opt.maxArgsForFlatMode;
  if (isKernel) {
    os << "[[kernel]] ";
  }
  os << "void " << F.getName().str() << " (";

  bool first = true;
  int bufIdx = 0;
  if (useArgStruct) {
    first = false;
    os << "device " << inputStructName << "& __args [[buffer(" << bufIdx++ << ")]]";
  } else {
    for (Argument &A : F.args()) {
      if (!first) os << ", ";
      first = false;
      auto typeName = mapType(A.getType());

      if (isKernel) {
        if (!A.getType()->isPointerTy()) {
          os << "constant " << typeName << "& " << valueName(&A) << " [[buffer(" << bufIdx++ << ")]]";
        } else {
          os << typeName << " " << valueName(&A) << " [[buffer(" << bufIdx++ << ")]]";
        }

      } else {
        os << typeName << " " << valueName(&A);
      }
    }
  }

  if (needDlm) {
    if (!first) {
      os << ", ";
    }

    if (isKernel) {
      os << "threadgroup void* __acpp_sscp_get_dynamic_local_memory [[threadgroup(0)]], ";
      os << "constant uint& __acpp_sscp_dynamic_local_memory_size [[buffer(" << bufIdx++ << ")]]";
    } else {
      os << "threadgroup void* __acpp_sscp_get_dynamic_local_memory, ";
      os << "uint __acpp_sscp_dynamic_local_memory_size";
    }
  }

  os << ")\n{\n";

  if (useArgStruct) {
    for (Argument &A : F.args()) {
      auto typeName = mapType(A.getType());
      os << indent(1) << typeName << " " << valueName(&A) << " = __args." << valueName(&A) << ";\n";
    }
    os << "\n";
  }
  return true;
}

bool MetalEmitter::emitDeclarations() {
  os << indent(1) << "// Locals\n";
  std::vector<std::pair<const Instruction*, int>> sortedAllocas(allocaIndex.begin(), allocaIndex.end());
  std::sort(sortedAllocas.begin(), sortedAllocas.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  for (auto &[AI, idx] : sortedAllocas) {
    Type *allocTy = dyn_cast<AllocaInst>(AI)->getAllocatedType();
    auto typeName = mapType(allocTy);
    os << indent(1) << typeName << " local" << idx << ";\n";
  }
  os << "\n";

  // PHI input variables - sorted by name
  if (!phiNodes.empty()) {
    os << indent(1) << "// PHI input variables\n";
    std::vector<const PHINode*> sortedPhis(phiNodes.begin(), phiNodes.end());
    std::unordered_map<const Value*, std::string> phiNameCache;
    for (const PHINode *PHI : sortedPhis) {
      phiNameCache[PHI] = valueName(PHI);
    }
    std::sort(sortedPhis.begin(), sortedPhis.end(),
              [&phiNameCache](const PHINode *a, const PHINode *b) { return phiNameCache[a] < phiNameCache[b]; });
    for (const PHINode *PHI : sortedPhis) {
      auto typeName = mapType(PHI);
      os << indent(1) << typeName << " " << phiNameCache[PHI] << "_in;\n";
    }
    os << "\n";
  }

  // All other variables - sorted by name
  os << indent(1) << "// Variables\n";
  std::vector<std::pair<const Value*, std::string>> sortedVars(valuesToDeclare.begin(), valuesToDeclare.end());
  std::unordered_map<const Value*, std::string> varNameCache;
  for (const auto &[V, ty] : sortedVars) {
    varNameCache[V] = valueName(V);
  }
  std::sort(sortedVars.begin(), sortedVars.end(),
            [&varNameCache](const auto &a, const auto &b) { return varNameCache[a.first] < varNameCache[b.first]; });
  for (auto &[V, ty] : sortedVars) {
    const std::string &name = varNameCache[V];
    // Skip allocas (handled above as pointers)
    if (isa<AllocaInst>(V)) {
      os << indent(1) << "thread void* " << name << ";\n";
      continue;
    }
    // Skip PHIs (handled separately)
    if (isa<PHINode>(V)) {
      os << indent(1) << mapType(V) << " " << name << ";\n";
      continue;
    }
    os << indent(1) << ty << " " << name << ";\n";
  }
  os << "\n";
  return true;
}

bool MetalEmitter::emitNode(const Node& node, int level) {
  bool success = true;
  switch (node.kind) {
    case NodeKind::List: {
      auto& listNode = static_cast<const ListNode&>(node);
      for (const auto& item : listNode.items) {
        if (item && success) {
          success &= emitNode(*item, level);
        }
      }
      break;
    }

    case NodeKind::Block: {
      auto& blockNode = static_cast<const BlockNode&>(node);
      if (blockNode.bb) {
        os << indent(level) << "// block " << basicBlockName(blockNode.bb) << "\n";
        success = emitBasicBlock(blockNode.bb, level);
      }
      break;
    }

    case NodeKind::If: {
      auto& ifNode = static_cast<const IfNode&>(node);

      os << indent(level) << "if (" << emitExpr(ifNode.cond) << ") {\n";
      if (ifNode.then_branch) {
        success &= emitNode(*ifNode.then_branch, level + 1);
      }
      os << indent(level) << "}";
      if (ifNode.else_branch) {
        os << " else {\n";
        success &= emitNode(*ifNode.else_branch, level + 1);
        os << indent(level) << "}";
      }
      os << "\n";
      break;
    }

    case NodeKind::Loop: {
      auto& loopNode = static_cast<const LoopNode&>(node);

      os << indent(level) << "while (true) {\n";
      if (loopNode.body) {
        success = emitNode(*loopNode.body, level + 1);
      }
      os << indent(level) << "}\n";

      break;
    }

    case NodeKind::Break: {
      os << indent(level) << "break;\n";
      break;
    }

    case NodeKind::Continue: {
      os << indent(level) << "continue;\n";
      break;
    }

    case NodeKind::Return: {
      auto& returnNode = static_cast<const ReturnNode&>(node);
      if (returnNode.value) {
        os << indent(level) << "return " << emitExpr(returnNode.value) << ";\n";
      } else {
        os << indent(level) << "return;\n";
      }
      break;
    }

    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown HL node kind: " << static_cast<int>(node.kind);
      errorMsg = ss.str();
      return false;
    }
  }
  return success;
}

bool MetalEmitter::emitBasicBlock(const BasicBlock* BB, int level) {
  const Instruction *TI = BB->getTerminator();
  bool success = true;
  for (const Instruction& I : *BB) {
    if (&I == TI) {
      continue;
    }
    if (!success) {
      return false;
    }
    success &= emitInstruction(I, level);
  }

  for (const PHINode *PHI : phiNodes) {
    auto it = phiIncomingFromBlock.find({BB, PHI});
    if (it != phiIncomingFromBlock.end()) {
      const Value *V = it->second;
      if (!isa<UndefValue>(V) && !isa<PoisonValue>(V)) {
        os << indent(level) << valueName(PHI) << "_in = " << emitExpr(V) << ";\n";
      } else {
        os << indent(level) << valueName(PHI) << "_in = /* undef */ 0;\n";
      }
    }
  }
  return success;
}

bool MetalEmitter::emitInstruction(const Instruction& I, int level) {
  std::string name = valueName(&I);

  if (auto *AI = dyn_cast<AllocaInst>(&I)) {
    os << indent(level) << name << " = &local" << allocaIndex[&I] << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    std::string ptr = emitExpr(LI->getPointerOperand());
    std::string elemType = mapType(LI->getType());
    std::string addrSpace;

    if (LI->getPointerOperandType()->isPointerTy()) {
      // Do NOT trust pointer operand type if it went through addrspacecast-to-generic.
      unsigned physAS = getPhysicalPointerAddressSpace(LI->getPointerOperand());
      addrSpace = getAddressSpaceKeyword(physAS);
    } else {
      errorMsg = "Error: Load from non-pointer type";
      return false;
    }

    if (LI->getType()->isPointerTy()) {
      // ** type, need to use struct slot hack to dereference pointer to pointer
      auto elemPhysAS = getPhysicalPointerAddressSpace(LI);
      auto elemAddrSpace = getAddressSpaceKeyword(elemPhysAS);
      auto structName = "__struct_ptr_to_" + elemAddrSpace;
      os << indent(level) << name << " = ((" << addrSpace << " " << structName << "*)" << ptr << ")->value; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << name << " = *((" << addrSpace << " " << elemType << "*)" << ptr << "); " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    std::string ptr = emitExpr(SI->getPointerOperand());
    std::string val = emitExpr(SI->getValueOperand());
    std::string elemType = mapType(SI->getValueOperand()->getType());
    std::string addrSpace;

    if (SI->getPointerOperandType()->isPointerTy()) {
      // Do NOT trust pointer operand type if it went through addrspacecast-to-generic.
      unsigned physAS = getPhysicalPointerAddressSpace(SI->getPointerOperand());
      addrSpace = getAddressSpaceKeyword(physAS);
    } else {
      errorMsg = "Error: Store to non-pointer type";
      return false;
    }

    if (SI->getValueOperand()->getType()->isPointerTy()) {
      // ** type, need to use struct slot hack to dereference pointer to pointer
      auto elemPhysAS = getPhysicalPointerAddressSpace(SI->getValueOperand());
      auto elemAddrSpace = getAddressSpaceKeyword(elemPhysAS);
      auto structName = "__struct_ptr_to_" + elemAddrSpace;
      os << indent(level) << "((" << addrSpace << " " << structName << "*)" << ptr << ")->value = " << val << "; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << "*((" << addrSpace << " " << elemType << "*)" << ptr << ") = " << val << "; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *PHI = dyn_cast<PHINode>(&I)) {
    os << indent(level) << name << " = " << name << "_in; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *UO = dyn_cast<UnaryOperator>(&I)) {
    emitUnaryOperator(UO, name, level);
    return true;
  }

  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    emitBinaryOperator(BO, name, level);
    return true;
  }

  if (auto *IC = dyn_cast<ICmpInst>(&I)) {
    emitICmpInstruction(IC, name, level);
    return true;
  }

  if (auto *FC = dyn_cast<FCmpInst>(&I)) {
    emitFCmpInstruction(FC, name, level);
    return true;
  }

  if (auto *SI = dyn_cast<SelectInst>(&I)) {
    std::string cond = emitExpr(SI->getCondition());
    std::string trueVal = emitExpr(SI->getTrueValue());
    std::string falseVal = emitExpr(SI->getFalseValue());
    os << indent(level) << name << " = (" << cond << ") ? (" << trueVal << ") : (" << falseVal << "); " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *CI = dyn_cast<CastInst>(&I)) {
    auto destType = mapType(CI->getDestTy());
    std::string src = emitExpr(CI->getOperand(0));
    auto srcType = mapType(CI->getSrcTy());
    // Special-case: addrspacecast to generic loses the information; in MSL we treat it as no-op.
    if (auto* ASC = dyn_cast<AddrSpaceCastInst>(CI)) {
      os << indent(level) << name << " = " << src << "; " << "// " << instToString(I) << "\n";
    } else if (auto* BC = dyn_cast<BitCastInst>(CI)) {
      os << indent(level) << name << " = " << "as_type<" << destType << ">(" << src << "); " << "// " << instToString(I) << "\n";
    } else if (auto* SI = dyn_cast<SExtInst>(CI)) {
      auto destSigned = getSignedType(CI->getDestTy());
      os << indent(level) << name << " = as_type<" << destType << ">((" << destSigned << ")__as_signed(" << src << ")); " << "// " << instToString(I) << "\n";
    } else if (srcType == "uint4" && destType == "uint") {
      os << indent(level) << name << " = " << src << ".x; " << "// " << instToString(I) << "\n";
    } else if (srcType == "uint" && destType == "uint4") {
      os << indent(level) << name << " = uint4(" << src << ", 0, 0, 0); " << "// " << instToString(I) << "\n";
    } else if (srcType == "uint4" || destType == "uint4") {
      errorMsg = "Error: Unsupported cast involving uint4 type";
      return false;
    } else {
      os << indent(level) << name << " = (" << destType << ") " << src << "; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    emitGEPInstruction(GEP, name, level);
    return true;
  }

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    return emitCallInstruction(CI, name, level);
  }

  if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
    std::string aggr = emitExpr(EV->getAggregateOperand());
    os << indent(level) << name << " = " << aggr;
    Type* aggrType = EV->getAggregateOperand()->getType();
    for (unsigned idx : EV->getIndices()) {
      if (aggrType->isStructTy()) {
        os << ".field" << idx;
        aggrType = aggrType->getStructElementType(idx);
      } else if (aggrType->isArrayTy()) {
        os << "[" << idx << "]";
        aggrType = aggrType->getArrayElementType();
      } else {
        errorMsg = "Error: ExtractValue from unsupported aggregate type";
        return false;
      }
    }
    os << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
    std::string aggr = emitExpr(IV->getAggregateOperand());
    std::string val = emitExpr(IV->getInsertedValueOperand());
    Type* aggrType = IV->getAggregateOperand()->getType();
    os << indent(level) << aggr;
    for (unsigned idx : IV->getIndices()) {
      if (aggrType->isStructTy()) {
        os << ".field" << idx;
        aggrType = aggrType->getStructElementType(idx);
      } else if (aggrType->isArrayTy()) {
        os << "[" << idx << "]";
        aggrType = aggrType->getArrayElementType();
      } else {
        errorMsg = "Error: InsertValue into unsupported aggregate type";
        return false;
      }
      os << " = " << val << "; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (Value *RV = RI->getReturnValue()) {
      os << indent(level) << "return " << emitExpr(RV) << "; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << "return; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *FI = dyn_cast<FreezeInst>(&I)) {
    std::string operand = emitExpr(FI->getOperand(0));
    os << indent(level) << name << " = " << operand << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  errorMsg = "ERROR: Unknown LLVM instruction: " + std::string(I.getOpcodeName());
  if (I.hasName()) {
    *errorMsg += " (name: " + I.getName().str() + ")";
  }
  *errorMsg += "\n";
  *errorMsg += "  " + instToString(I) + "\n";
  return false;
}

void MetalEmitter::emitUnaryOperator(const UnaryOperator* UO, const std::string& name, int level) {
  switch (UO->getOpcode()) {
    case Instruction::FNeg: {
      std::string operand = emitExpr(UO->getOperand(0));
      os << indent(level) << name << " = -" << operand << "; " << "// " << instToString(*UO) << "\n";
      break;
    }

    default:
      errorMsg = "ERROR: Unknown unary operator: " + std::string(UO->getOpcodeName());
      return;
  }
}

void MetalEmitter::emitBinaryOperator(const BinaryOperator* BO, const std::string& name, int level) {
  std::string lhs = emitExpr(BO->getOperand(0));
  std::string rhs = emitExpr(BO->getOperand(1));

  std::string resultType = mapType(BO->getType());

  switch (BO->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::Add:
      os << indent(level) << name << " = " << lhs << " + " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FSub:
    case Instruction::Sub:
      os << indent(level) << name << " = " << lhs << " - " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FMul:
    case Instruction::Mul:
      os << indent(level) << name << " = " << lhs << " * " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FDiv:
    case Instruction::UDiv:
      os << indent(level) << name << " = " << lhs << " / " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;
    case Instruction::URem:
      os << indent(level) << name << " = " << lhs << " % " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::SDiv:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) / (__as_signed(" << rhs << ")));\n";
      break;
    case Instruction::SRem:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) % (__as_signed(" << rhs << ")));\n";
      break;
    case Instruction::FRem:
      os << indent(level) << name << " = fmod(" << lhs << ", " << rhs << ");\n";
      break;

    case Instruction::Shl:
      if (resultType == "uint4") {
        // TODO: track >> + trunc for performance
        auto _x = lhs + ".x";
        auto _y = lhs + ".y";
        auto _z = lhs + ".z";
        if (rhs == "0x20u") {
          os << indent(level) << name << " = uint4(0," << _x << "," << _y << "," << _z << "); //" << instToString(*BO) << "\n";
        } else if (rhs == "0x40u") {
          os << indent(level) << name << " = uint4(0,0," << _x << "," << _y << "); //" << instToString(*BO) << "\n";
        } else if (rhs == "0x60u") {
          os << indent(level) << name << " = uint4(0,0,0," << _x << "); //" << instToString(*BO) << "\n";
        } else {
          errorMsg = "ERROR: Unsupported uint4 shift left amount: " + rhs;
        }
      } else {
        os << indent(level) << name << " = " << lhs << " << " << rhs << ";\n";
      }
      break;
    case Instruction::AShr:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) >> (__as_signed(" << rhs << ")));\n";
      break;

    case Instruction::LShr:
      if (resultType == "uint4") {
        // TODO: track >> + trunc for performance
        auto _y = lhs + ".y";
        auto _z = lhs + ".z";
        auto _w = lhs + ".w";
        if (rhs == "0x20u") {
          os << indent(level) << name << " = uint4(" << _y << "," << _z << "," << _w << ",0); // " << instToString(*BO) << "\n";
        } else if (rhs == "0x40u") {
          os << indent(level) << name << " = uint4(" << _z << "," << _w << ",0,0); // " << instToString(*BO) << "\n";
        } else if (rhs == "0x60u") {
          os << indent(level) << name << " = uint4(" << _w << ",0,0,0); // " << instToString(*BO) << "\n";
        } else {
          errorMsg = "ERROR: Unsupported uint4 logical right shift amount: " + rhs;
        }
      } else {
        os << indent(level) << name << " = " << lhs << " >> " << rhs << ";\n";
      }
      break;

    // bit logical ops
    case Instruction::And:
      os << indent(level) << name << " = " << lhs << " & " << rhs << ";\n";
      break;
    case Instruction::Or:
      os << indent(level) << name << " = " << lhs << " | " << rhs << ";\n";
      break;
    case Instruction::Xor:
      os << indent(level) << name << " = " << lhs << " ^ " << rhs << ";\n";
      break;


    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown binary operator: " << BO->getOpcodeName() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitICmpInstruction(const ICmpInst* IC, const std::string& name, int level) {
  std::string lhs = emitExpr(IC->getOperand(0));
  std::string rhs = emitExpr(IC->getOperand(1));

  auto resultType = mapType(IC->getType());

  switch (IC->getPredicate()) {
    case ICmpInst::ICMP_EQ:
      os << indent(level) << name << " = (" << lhs << " == " << rhs << ");\n";
      break;
    case ICmpInst::ICMP_NE:
      os << indent(level) << name << " = (" << lhs << " != " << rhs << ");\n";
      break;
    case ICmpInst::ICMP_UGT:
      os << indent(level) << name << " = (" << lhs << " > " << rhs << ");\n";
      break;
    case ICmpInst::ICMP_UGE:
      os << indent(level) << name << " = (" << lhs << " >= " << rhs << ");\n";
      break;
    case ICmpInst::ICMP_ULT:
      os << indent(level) << name << " = (" << lhs << " < " << rhs << ");\n";
      break;
    case ICmpInst::ICMP_ULE:
      os << indent(level) << name << " = (" << lhs << " <= " << rhs << ");\n";
      break;

    // signed variants
    case ICmpInst::ICMP_SGT:
      os << indent(level) << name << " = ((__as_signed(" << lhs << ")) > (__as_signed(" << rhs << ")));\n";
      break;
    case ICmpInst::ICMP_SGE:
      os << indent(level) << name << " = ((__as_signed(" << lhs << ")) >= (__as_signed(" << rhs << ")));\n";
      break;
    case ICmpInst::ICMP_SLT:
      os << indent(level) << name << " = ((__as_signed(" << lhs << ")) < (__as_signed(" << rhs << ")));\n";
      break;
    case ICmpInst::ICMP_SLE:
      os << indent(level) << name << " = ((__as_signed(" << lhs << ")) <= (__as_signed(" << rhs << ")));\n";
      break;
    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown ICmp predicate: " << IC->getPredicate() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitFCmpInstruction(const FCmpInst* FC, const std::string& name, int level) {
  std::string lhs = emitExpr(FC->getOperand(0));
  std::string rhs = emitExpr(FC->getOperand(1));

  switch (FC->getPredicate()) {
    case FCmpInst::FCMP_OEQ:
      os << indent(level) << name << " = (" << lhs << " == " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_ONE:
      os << indent(level) << name << " = (" << lhs << " != " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OGT:
      os << indent(level) << name << " = (" << lhs << " > " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OGE:
      os << indent(level) << name << " = (" << lhs << " >= " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OLT:
      os << indent(level) << name << " = (" << lhs << " < " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OLE:
      os << indent(level) << name << " = (" << lhs << " <= " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_ORD:
      os << indent(level) << name << " = isordered(" << lhs << ", " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_UNO:
      os << indent(level) << name << " = isunordered(" << lhs << ", " << rhs << ");\n";
      break;

    case FCmpInst::FCMP_UEQ:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " == " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UGT:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " > " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UGE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " >= " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_ULT:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " < " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_ULE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " <= " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UNE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " != " << rhs << "));\n";
      break;

    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown FCmp predicate: " << FC->getPredicate() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitGEPInstruction(const GetElementPtrInst* GEP, const std::string& name, int level) {
  std::string base = emitExpr(GEP->getPointerOperand());
  Type* srcElemType = GEP->getSourceElementType();
  std::string typeName = mapType(srcElemType);

  unsigned physAS = getPhysicalPointerAddressSpace(GEP->getPointerOperand());
  std::string addrSpace = getAddressSpaceKeyword(physAS);

  std::string accessExpr;
  Type *currentType = srcElemType;

  auto idxIt = GEP->idx_begin();
  if (idxIt != GEP->idx_end()) {
    Value *firstIdx = *idxIt;
    std::string castBase = "((" + addrSpace + " " + typeName + "*)" + base + ")";

    if (auto *CI = dyn_cast<ConstantInt>(firstIdx)) {
      accessExpr = castBase + "[" + std::to_string(CI->getSExtValue()) + "]";
    } else {
      accessExpr = castBase + "[" + emitExpr(firstIdx) + "]";
    }
    ++idxIt;
  } else {
    // No indices - just cast
    accessExpr = "((" + addrSpace + " " + typeName + "*)" + base + ")";
  }

  for (; idxIt != GEP->idx_end(); ++idxIt) {
    Value *idx = *idxIt;

    if (auto *ST = dyn_cast<StructType>(currentType)) {
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        unsigned fieldIdx = CI->getZExtValue();
        accessExpr += ".field" + std::to_string(fieldIdx);
        currentType = ST->getElementType(fieldIdx);
      } else {
        errorMsg = "Error: Non-constant struct index in GEP\n";
        return;
      }
    } else if (auto *AT = dyn_cast<ArrayType>(currentType)) {
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        accessExpr += "[" + std::to_string(CI->getZExtValue()) + "]";
      } else {
        accessExpr += "[" + emitExpr(idx) + "]";
      }
      currentType = AT->getElementType();
    } else {
      // Pointer arithmetic for other types
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        if (!CI->isZero()) {
          accessExpr = "(" + accessExpr + " + " + std::to_string(CI->getSExtValue()) + ")";
        }
      } else {
        accessExpr = "(" + accessExpr + " + " + emitExpr(idx) + ")";
      }
    }
  }

  os << indent(level) << name << " = &(" << accessExpr << "); " << "// " << instToString(*GEP) << "\n";
}

bool MetalEmitter::emitCallInstruction(const CallInst* CI, const std::string& name, int level) {
  Function *callee = CI->getCalledFunction();

  std::string calleeName = callee->getName().str();

  int argsSize = CI->arg_size();
  if (callee->isDeclaration()) {
    auto* funcInfo = externalFunctionMapper.getFunctionInfo(calleeName);
    if (!funcInfo) {
      errorMsg = "Error: No mapping found for external function: " + calleeName + "\n";
      return false;
    }
    if (funcInfo->ignore) {
      // function is ignored - do nothing
      return true;
    }
    if (funcInfo->customCallEmitter) {
      std::string errorStr;
      auto callExpr = funcInfo->customCallEmitter(CI, errorStr);
      if (!errorStr.empty()) {
        errorMsg = errorStr;
        return false;
      }
      if (!CI->getType()->isVoidTy()) {
        os << indent(level) << name << " = " << *callExpr << ";\n";
      } else {
        os << indent(level) << *callExpr << ";\n";
      }
      return true;
    }

    calleeName = funcInfo->replacement ? *funcInfo->replacement : funcInfo->name;
    if (funcInfo->convertToVar) {
      // treat as variable
      os << indent(level) << name << " = " << calleeName << ";\n";
      return true;
    }
    auto returnType = callee->getReturnType();
    if (returnType->isStructTy()) {
      calleeName += "<" + mapType(returnType) + ">";
    }
    if (funcInfo->argsCount >= 0) {
      argsSize = funcInfo->argsCount;
    }
  }

  std::string callExpr = calleeName + "(";
  for (unsigned i = 0; i < argsSize; ++i) {
    if (i > 0) {
      callExpr += ", ";
    }
    callExpr += emitExpr(CI->getArgOperand(i));
  }
  if (needsDynamicLocalMemory.count(callee)) {
    if (argsSize > 0) {
      callExpr += ", ";
    }
    callExpr += "__acpp_sscp_get_dynamic_local_memory, ";
    callExpr += "__acpp_sscp_dynamic_local_memory_size";
  }
  callExpr += ")";

  if (!CI->getType()->isVoidTy()) {
    os << indent(level) << name << " = " << callExpr << ";\n";
  } else {
    os << indent(level) << callExpr << ";\n";
  }
  return true;
}

std::string MetalEmitter::emitExpr(const Value* V) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getBitWidth() == 1) {
      return CI->isZero() ? "false" : "true";
    }
    uint64_t val = CI->getZExtValue();
    std::ostringstream hex;
    hex << "0x" << std::hex << val << "u";
    if (CI->getBitWidth() == 64) {
      hex << "l";
    }
    return hex.str();
  }

  if (auto *CF = dyn_cast<ConstantFP>(V)) {
    float floatVal = CF->getValue().convertToFloat();
    uint32_t val; memcpy(&val, &floatVal, sizeof(float));
    std::ostringstream hex;
    hex << "as_type<float>(" << "0x" << std::hex << val << "u)";
    return hex.str();
  }

  if (isa<ConstantPointerNull>(V)) {
    return "nullptr";
  }

  if (isa<UndefValue>(V) || isa<PoisonValue>(V)) {
    return "/* undef */ 0";
  }

  return valueName(V);
}

std::string MetalEmitter::valueName(const Value* V) {
  std::string s;
  raw_string_ostream rso(s);
  V->printAsOperand(rso, false);
  std::string name = rso.str();
  if (name[0] == '%') {
    name = name.substr(1);
  }
  for (char& c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      c = '_';
    }
  }
  return "t" + name;
}

std::string MetalEmitter::basicBlockName(const BasicBlock* BB) {
  if (BB->getName().str().empty()) {
    std::string s;
    raw_string_ostream rso(s);
    BB->printAsOperand(rso, false);
    return rso.str();
  } else {
    return "%" + BB->getName().str();
  }
}

std::string MetalEmitter::getSignedType(Type* T) {
  if (T->isIntegerTy()) {
    unsigned bitWidth = T->getIntegerBitWidth();
    if (bitWidth == 1) {
      return "bool";
    } else if (bitWidth == 8) {
      return "char";
    } else if (bitWidth == 16) {
      return "short";
    } else if (bitWidth == 32) {
      return "int";
    } else if (bitWidth == 64) {
      return "long";
    } else if (bitWidth == 128) {
      return "int4";
    } else {
      std::ostringstream ss;
      ss << "Error: Unsupported integer bit width: " << bitWidth << "\n";
      errorMsg = ss.str();
      return "";
    }
  } else {
    errorMsg = "Error: getSignedType called on non-integer type\n";
    return "";
  }
}

std::string MetalEmitter::mapType(const Type* T) {
  auto it = typeCache.find(T);
  if (it != typeCache.end()) {
    return it->second;
  }

  if (T->isPointerTy()) {
    unsigned AS = T->getPointerAddressSpace();
    auto keyword = getAddressSpaceKeyword(AS);

    return typeCache[T] = keyword + " void*";
  }

  if (auto *AT = dyn_cast<ArrayType>(T)) {
    auto elemType = mapType(AT->getElementType());
    return typeCache[T] = "array<" + elemType + ", " + std::to_string(AT->getNumElements()) + ">";
  }

  if (auto *ST = dyn_cast<StructType>(T)) {
    if (ST->hasName()) {
      auto str = ST->getName().str();
      for (char& c : str) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
          c = '_';
        }
      }
      return typeCache[T] = str;
    }

    auto structName = "anon_struct_" + std::to_string(nextAnonymousStructId++);
    anonStructs[T] = structName;
    return typeCache[T] = structName;
  }

  if (T->isVoidTy()) {
    return typeCache[T] = "void";
  } else if (T->isIntegerTy()) {
    unsigned bitWidth = T->getIntegerBitWidth();
    if (bitWidth == 1) {
      return typeCache[T] = "bool";
    } else if (bitWidth == 8) {
      return typeCache[T] = "uchar";
    } else if (bitWidth == 16) {
      return typeCache[T] = "ushort";
    } else if (bitWidth == 32) {
      return typeCache[T] = "uint";
    } else if (bitWidth == 48) {
      auto* info = externalFunctionMapper.getFunctionInfo("i48u"); // mark used
      if (!info) {
        errorMsg = "Error: i48u type used but no mapping found\n";
        return "";
      }
      return typeCache[T] = "i48u";
    } else if (bitWidth == 64) {
      return typeCache[T] = "ulong";
    } else if (bitWidth == 128) {
      return typeCache[T] = "uint4";
    } else {
      std::ostringstream ss;
      ss << "Error: Unsupported integer bit width: " << bitWidth << "\n";
      errorMsg = ss.str();
      return "";
    }
  } else if (T->isFloatTy()) {
    return typeCache[T] = "float";
  } else if (T->isDoubleTy()) {
    errorMsg = "Error: Double type is not supported on Metal GPU\n";
    return "";
  } else if (T->isHalfTy()) {
    return typeCache[T] = "half";
  } else {
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    T->print(rso);
    rso.flush();
    errorMsg = "Error: Unsupported type encountered : " + typeStr;
    return "";
  }
}

std::string MetalEmitter::mapType(const llvm::Value* V) {
  if (V->getType()->isPointerTy()) {
    unsigned AS = getPhysicalPointerAddressSpace(V);
    auto keyword = getAddressSpaceKeyword(AS);
    return keyword + " void*";
  } else {
    return mapType(V->getType());
  }
}

std::string MetalEmitter::getAddressSpaceKeyword(unsigned AS) {
  auto it = addressSpaceMap.find(AS);
  if (it != addressSpaceMap.end()) {
    return it->second;
  } else {
    errorMsg = "Error: Unknown address space " + std::to_string(AS) + "\n";
    return "";
  }
}

const Value* MetalEmitter::stripToRootObject(const Value* V) {
  const Value* Cur = V;
  while (true) {
    if (auto* ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
      Cur = ASC->getPointerOperand();
      continue;
    }
    if (auto* BC = dyn_cast<BitCastInst>(Cur)) {
      Cur = BC->getOperand(0);
      continue;
    }
    if (auto* GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      // Accept only "trivial" GEPs (all indices are zero) for minimal patch
      bool allZero = true;
      for (auto it = GEP->idx_begin(); it != GEP->idx_end(); ++it) {
        if (auto* CI = dyn_cast<ConstantInt>(*it)) {
          if (!CI->isZero()) { allZero = false; break; }
        } else {
          allZero = false;
          break;
        }
      }
      if (allZero) {
        Cur = GEP->getPointerOperand();
        continue;
      }
    }
    break;
  }
  return Cur;
}

unsigned MetalEmitter::getPhysicalPointerAddressSpace(const Value* V) {
  Type* T = V->getType();
  if (!T->isPointerTy()) {
    return 0; // not a pointer
  }
  unsigned AS = T->getPointerAddressSpace();
  if (AS != 0) {
    return AS;
  }
  auto it = inferredPtrAS.find(V);
  if (it != inferredPtrAS.end()) {
    return it->second;
  }

  // If the value is an addrspacecast, get the original address space
  if (auto* ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    return (inferredPtrAS[V] = getPhysicalPointerAddressSpace(ASC->getOperand(0)));
  }
  if (auto* GEP = dyn_cast<GetElementPtrInst>(V)) {
    return (inferredPtrAS[V] = getPhysicalPointerAddressSpace(GEP->getPointerOperand()));
  }
  if (auto* PHI = dyn_cast<PHINode>(V)) {
    unsigned deducedAS = 0;
    for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
      Value* incomingV = PHI->getIncomingValue(i);
      unsigned incomingAS = getPhysicalPointerAddressSpace(incomingV);
      if (incomingAS != 0) {
        if (deducedAS == 0) {
          deducedAS = incomingAS;
        } else if (deducedAS != incomingAS) {
          errorMsg = "Error: Conflicting address space information in PHI node\n";
          return 0;
        }
      }
    }
    if (deducedAS != 0) {
      return (inferredPtrAS[V] = deducedAS);
    }
  }
  if (auto* Select = dyn_cast<SelectInst>(V)) {
    unsigned trueAS = getPhysicalPointerAddressSpace(Select->getTrueValue());
    unsigned falseAS = getPhysicalPointerAddressSpace(Select->getFalseValue());
    if (trueAS != 0 && trueAS == falseAS) {
      return (inferredPtrAS[V] = trueAS);
    }
  }
  if (auto* LI = dyn_cast<LoadInst>(V)) {
    const Value* addr = LI->getPointerOperand();       // address of the slot
    const Value* root = stripToRootObject(addr);       // e.g. %.sroa.xxx.i alloca in AS5

    unsigned deducedAS = 0;

    // Scan stores that write into the same root object
    for (const User* U : root->users()) {
      const StoreInst* SI = dyn_cast<StoreInst>(U);
      if (!SI) continue;

      // Does this store target the same root (allow casts/trivial GEP)?
      const Value* stRoot = stripToRootObject(SI->getPointerOperand());
      if (stRoot != root) continue;

      unsigned storedAS = getPhysicalPointerAddressSpace(SI->getValueOperand());
      if (storedAS == 0) continue;

      if (deducedAS == 0) deducedAS = storedAS;
      else if (deducedAS != storedAS) {
        // Conflicting stores into same slot
        return 0;
      }
    }

    if (deducedAS != 0) {
      return (inferredPtrAS[V] = deducedAS);
    }
  }

  return 0;
}

void MetalEmitter::analyzeCallInsts() {
  std::unordered_map<const Function*, std::vector<const Function*>> calls;

  for (auto& F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    for (BasicBlock& BB : F) {
      for (Instruction& I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) {
          continue;
        }
        Function *Callee = CI->getCalledFunction();
        if (!Callee) {
          continue;
        }
        bool ignoreCallee = false;
        if (Callee->isDeclaration()) {
          auto* info = externalFunctionMapper.getFunctionInfo(Callee->getName(). str());
          if (info && info->ignore) {
            ignoreCallee = true;
          }
          if (info && info->needsLocalMemory) {
            needsDynamicLocalMemory.insert(&F);
            needsDynamicLocalMemory.insert(Callee);
          }
        }
        if (ignoreCallee) {
          continue;
        }
        // add return type and args to cache
        auto returnType = Callee->getReturnType();
        mapType(returnType);
        for (auto& Arg : Callee->args()) {
          mapType(Arg.getType());
        }
        const StringRef calleeName = Callee->getName();
        if (calleeName == "__acpp_sscp_get_dynamic_local_memory") {
          needsDynamicLocalMemory.insert(&F);
        } else if (!Callee->isDeclaration()) {
          calls[&F].push_back(Callee);
        }
      }
    }
  }

  // transitive closure, TODO: union-find
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto& [F, callees] : calls) {
      if (needsDynamicLocalMemory.count(F)) {
        continue;
      }
      for (auto* Callee : callees) {
        if (needsDynamicLocalMemory.count(Callee)) {
          needsDynamicLocalMemory.insert(F);
          changed = true;
          break;
        }
      }
    }
  }
}

void MetalEmitter::collectVariablesInfo(const Function& F) {
  allocaIndex.clear();
  phiSources.clear();
  sourceToPhis.clear();
  phiIncomingFromBlock.clear();
  phiNodes.clear();
  valuesToDeclare.clear();

  for (const BasicBlock& BB : F) {
    for (const Instruction& I : BB) {
      if (auto* AI = dyn_cast<AllocaInst>(&I)) {
        int index = static_cast<int>(allocaIndex.size());
        allocaIndex[&I] = index;
      }

      if (auto* PHI = dyn_cast<PHINode>(&I)) {
        phiNodes.insert(PHI);
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
          Value* V = PHI->getIncomingValue(i);
          BasicBlock* incomingBB = PHI->getIncomingBlock(i);

          phiSources[PHI].push_back(V);
          phiIncomingFromBlock[{incomingBB, PHI}] = V;

          if (!isa<Constant>(V)) {
            sourceToPhis[V].push_back(PHI);
          }
        }
      }

      if (!I.getType()->isVoidTy()) {
        if (auto* ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
          valuesToDeclare[&I] = mapType(ASC->getOperand(0));
        } else if (auto* GEP = dyn_cast<GetElementPtrInst>(&I)) {
          valuesToDeclare[&I] = mapType(GEP->getPointerOperand());
        } else {
          valuesToDeclare[&I] = mapType(&I);
        }
      }
    }
  }
}

} // namespace compiler
} // namespace hipsycl
