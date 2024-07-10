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
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &Ost, const llvm::StringRef &StrRef) {
  return Ost << StrRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationInfo::analyzeModule(llvm::Module &M) {
  if (auto *BarrIntrinsic = M.getFunction(hipsycl::compiler::cbs::BarrierIntrinsicName)) {
    SplitterFuncs.insert(BarrIntrinsic);
    HIPSYCL_DEBUG_INFO << "Found splitter intrinsic " << BarrIntrinsic->getName() << "\n";
  }

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (Annotation.compare(SplitterAnnotation) == 0) {
      SplitterFuncs.insert(F);
      HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << F->getName() << "\n";
    } else if (Annotation.compare(KernelAnnotation) == 0) {
      NDKernels.insert(F);
      HIPSYCL_DEBUG_INFO << "Found kernel annotated function " << F->getName() << "\n";
    }
  });

  if (auto MD = M.getNamedMetadata(SscpAnnotationsName))
    for (auto OP : MD->operands()) {
      if (OP->getNumOperands() == 3 &&
          llvm::cast<llvm::MDString>(OP->getOperand(1))->getString() == SSCPKernelMD &&
          llvm::cast<llvm::ConstantInt>(
              llvm::cast<llvm::ConstantAsMetadata>(OP->getOperand(2))->getValue())
                  ->getZExtValue() == 1)
        NDKernels.insert(llvm::cast<llvm::Function>(
            llvm::cast<llvm::ValueAsMetadata>(OP->getOperand(0))->getValue()));
    }
  return false;
}

hipsycl::compiler::SplitterAnnotationInfo::SplitterAnnotationInfo(llvm::Module &Module) {
  analyzeModule(Module);
}

void hipsycl::compiler::SplitterAnnotationInfo::print(llvm::raw_ostream &Stream) {
  Stream << "Splitters:\n";
  for (auto *F : SplitterFuncs) {
    Stream << F->getName() << "\n";
  }
  Stream << "NDRange Kernels:\n";
  for (auto *F : NDKernels) {
    Stream << F->getName() << "\n";
  }
}

bool hipsycl::compiler::SplitterAnnotationAnalysisLegacy::runOnFunction(llvm::Function &F) {
  if (SplitterAnnotation_)
    return false;
  SplitterAnnotation_ = SplitterAnnotationInfo{*F.getParent()};
  return false;
}

hipsycl::compiler::SplitterAnnotationAnalysis::Result
hipsycl::compiler::SplitterAnnotationAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  return SplitterAnnotationInfo{M};
}

char hipsycl::compiler::SplitterAnnotationAnalysisLegacy::ID = 0;
llvm::AnalysisKey hipsycl::compiler::SplitterAnnotationAnalysis::Key;

llvm::PreservedAnalyses
hipsycl::compiler::SplitterAnnotationAnalysisCacher::run(llvm::Module &M,
                                                         llvm::ModuleAnalysisManager &AM) {
  AM.getResult<SplitterAnnotationAnalysis>(M);
  return llvm::PreservedAnalyses::all();
}
