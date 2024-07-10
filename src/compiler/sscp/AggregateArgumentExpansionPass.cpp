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
#include "hipSYCL/compiler/sscp/AggregateArgumentExpansionPass.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/utils/AggregateTypeUtils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>

namespace hipsycl {
namespace compiler {

namespace {


llvm::Type* getValueType(llvm::Function& F, int ArgNo) {
  llvm::Type* ArgT = F.getFunctionType()->getParamType(ArgNo);
  if(ArgT->isPointerTy() && F.hasParamAttribute(ArgNo, llvm::Attribute::ByVal)) {
    llvm::Type *ValT = F.getParamAttribute(ArgNo, llvm::Attribute::ByVal).getValueAsType();
    return ValT;
  } else {
    return ArgT;
  }
}

bool needsExpansion(llvm::Function& F, int ArgNo) {
  llvm::Type* ArgT = F.getFunctionType()->getParamType(ArgNo);
  if(ArgT->isPointerTy() && F.hasParamAttribute(ArgNo, llvm::Attribute::ByVal))
    return true;
  if(ArgT->isAggregateType())
    return true;
  return false;
}

struct ExpandedArgumentInfo {
  // Original value type
  llvm::Type* OriginalByValType = nullptr;
  
  // Indices for getelementptr to all expanded arguments of the original
  // arg
  llvm::SmallVector<llvm::SmallVector<int, 16>> GEPIndices;
  llvm::SmallVector<llvm::Type*> ExpandedTypes;
  llvm::SmallVector<llvm::SmallVector<std::string>> TypeAnnotations;
  // original argument index
  int OriginalIndex = 0;
  // Number of arguments the original arg was expanded into
  int NumExpandedArguments = 1;
  bool IsExpanded = false;
};

static const char* TypeAnnotationIdentifier = "__acpp_sscp_emit_param_type_annotation";

std::string ExtractAnnotationFromType(llvm::Type* T) {
  if(!T)
    return {};
  
  std::string StructName = T->getStructName().str();

  std::size_t pos = StructName.find(TypeAnnotationIdentifier);
  if(pos == std::string::npos)
    return {};

  std::string Substr = StructName.substr(pos);
  std::string AnnotationType = Substr.substr(0, Substr.find_first_of(":."));

  std::string Prefix = TypeAnnotationIdentifier;
  Prefix += "_";
  auto Result = AnnotationType.substr(Prefix.size());

  return Result;
}

void ExpandAggregateArguments(llvm::Module &M, llvm::Function &F,
                              std::vector<OriginalParamInfo> &OriginalParamInfos) {
  OriginalParamInfos.clear();

  // stores one entry per *original* function argument. Maps
  // original argument index to expansion info.
  llvm::SmallDenseMap<int, ExpandedArgumentInfo> ExpansionInfo;

  for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
    ExpandedArgumentInfo Info;
    Info.OriginalIndex = i;

    if (needsExpansion(F, i)) {
      Info.IsExpanded = true;

      auto OnContainedType = [&](llvm::Type *T, llvm::SmallVector<int, 16> Indices,
                                 llvm::SmallVector<llvm::Type *, 16> MatchedParentTypes) {
        Info.ExpandedTypes.push_back(T);
        Info.GEPIndices.push_back(Indices);
        
        llvm::SmallVector<std::string> Annotations;
        for(llvm::Type* MatchedType : MatchedParentTypes) {
          std::string Annotation = ExtractAnnotationFromType(MatchedType);
          if(!Annotation.empty()) {
            if (std::find(Annotations.begin(), Annotations.end(), Annotation) == Annotations.end()) {
              Annotations.push_back(Annotation);
            }
          }
        }

        Info.TypeAnnotations.push_back(Annotations);
      };

      auto ParentTypeMatcher = [&](llvm::Type* T) {
        if(T && T->isStructTy())
          if(T->getStructName().find(TypeAnnotationIdentifier))
            return true;
        return false;
      };

      auto* ValueT = getValueType(F, i);
      utils::ForEachNonAggregateContainedTypeWithParentTypeMatcher(ValueT, OnContainedType, {}, {},
                                                                   ParentTypeMatcher);
      Info.OriginalByValType = ValueT;
      Info.NumExpandedArguments = Info.GEPIndices.size();
    } else {
      auto* ArgT = F.getFunctionType()->getParamType(i);
      Info.OriginalByValType = ArgT;
    }

    ExpansionInfo[i] = Info;
  }

  llvm::SmallVector<llvm::Type*, 16> NewArgumentTypes;
  for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
    auto& EI = ExpansionInfo[i];
    if(EI.IsExpanded) {
      for(auto* T : EI.ExpandedTypes)
        NewArgumentTypes.push_back(T);
    } else {
      NewArgumentTypes.push_back(EI.OriginalByValType);
    }
  }

  std::string FunctionName = F.getName().str();
  F.setName(FunctionName + "_PreArgumentExpansion");
  auto OldLinkage = F.getLinkage();
  F.setLinkage(llvm::GlobalValue::InternalLinkage);

  llvm::FunctionType *FType = llvm::FunctionType::get(F.getReturnType(), NewArgumentTypes, false);
  if (auto *NewF = llvm::dyn_cast<llvm::Function>(
          M.getOrInsertFunction(FunctionName, FType).getCallee())) {
    for(auto& Attr : F.getAttributes().getFnAttrs()) {
      NewF->addFnAttr(Attr);
    }
    NewF->setLinkage(OldLinkage);

    if (auto Annotations = M.getNamedMetadata(SscpAnnotationsName)) {
      for (auto *MD : Annotations->operands()) {
        if (&F == llvm::cast<llvm::Function>(
                      llvm::cast<llvm::ValueAsMetadata>(MD->getOperand(0))->getValue())) {
          MD->replaceOperandWith(0, llvm::ValueAsMetadata::get(NewF));
        }
      }
    }

    llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", NewF);
    
    llvm::SmallVector<llvm::Value*, 8> CallArgs;

    int CurrentNewIndex = 0;
    for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
      auto& EI = ExpansionInfo[i];

      if(EI.IsExpanded) {
        // We need to reconstruct the aggregate that was decomposed.
        // First, allocate some memory
        auto* Alloca = new llvm::AllocaInst(EI.OriginalByValType, 0, "", BB);
        // Iterate over all elements that the aggregate was expanded into:
        for(int j = 0; j < EI.NumExpandedArguments; ++j) {
          // Create getelementptr instruction to offset into the allocated struct

          llvm::SmallVector<llvm::Value*> GEPIndices;
          GEPIndices.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 0));
          for(int Idx : EI.GEPIndices[j])
            GEPIndices.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), Idx));

          llvm::ArrayRef<llvm::Value *> GEPIndicesRef{GEPIndices};

          auto *GEPInst = llvm::GetElementPtrInst::CreateInBounds(
              EI.OriginalByValType, Alloca, llvm::ArrayRef<llvm::Value *>{GEPIndicesRef}, "", BB);
          // Store expanded argument into allocated space
          assert(CurrentNewIndex + j < NewF->getFunctionType()->getNumParams());

          auto *StoredVal = NewF->getArg(CurrentNewIndex + j);
          [[maybe_unused]] auto *StoreInst = new llvm::StoreInst(StoredVal, GEPInst, BB);

          // Store the indexed offset - runtimes can use this information later
          // when invoking the function.
          std::size_t IndexedOffset = M.getDataLayout().getIndexedOffsetInType(
              EI.OriginalByValType, GEPIndicesRef);
          OriginalParamInfos.push_back(OriginalParamInfo{
              IndexedOffset, static_cast<std::size_t>(EI.OriginalIndex), EI.TypeAnnotations[j]});
        }
        CallArgs.push_back(Alloca);
      } else {
        CallArgs.push_back(NewF->getArg(CurrentNewIndex));
        OriginalParamInfos.push_back(OriginalParamInfo{
            0, static_cast<std::size_t>(EI.OriginalIndex), EI.TypeAnnotations[0]});
      }

      CurrentNewIndex += EI.NumExpandedArguments;
    }
    
    auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(&F), CallArgs,
                                            "", BB);
    for(int i = 0; i < CallArgs.size(); ++i) {
      if(F.hasParamAttribute(i, llvm::Attribute::ByVal))
        Call->addParamAttr(i, F.getParamAttribute(i, llvm::Attribute::ByVal));
    }
    llvm::ReturnInst::Create(M.getContext(), BB);

    if(!F.hasFnAttribute(llvm::Attribute::AlwaysInline))
      F.addFnAttr(llvm::Attribute::AlwaysInline);
  }
}
}

AggregateArgumentExpansionPass::AggregateArgumentExpansionPass(
    const std::vector<std::string> &FunctioNames)
    : AffectedFunctionNames{FunctioNames} {}

llvm::PreservedAnalyses AggregateArgumentExpansionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  OriginalParamInfos.clear();
  for(const auto& FN : AffectedFunctionNames) {
    if(auto* F = M.getFunction(FN)) {
      std::vector<OriginalParamInfo> OIs;
      ExpandAggregateArguments(M, *F, OIs);
      this->OriginalParamInfos[FN] = OIs;
    }
  }
  return llvm::PreservedAnalyses::none();
}

const std::vector<OriginalParamInfo>*
AggregateArgumentExpansionPass::getInfosOnOriginalParams(const std::string &FunctionName) const {
  auto it = OriginalParamInfos.find(FunctionName);
  if(it == OriginalParamInfos.end())
    return nullptr;
  return &(it->second);
}
}
}
