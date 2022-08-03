/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Constants.h>

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &Ost, const llvm::StringRef &StrRef) {
  return Ost << StrRef.begin();
}

bool hipsycl::compiler::SplitterAnnotationInfo::analyzeModule(llvm::Module &M) {
  if (auto *BarrIntrinsic = M.getFunction(hipsycl::compiler::BarrierIntrinsicName)) {
    SplitterFuncs.insert(BarrIntrinsic);
    HIPSYCL_DEBUG_INFO << "Found splitter intrinsic " << BarrIntrinsic->getName() << "\n";
  }

  for (auto &I : M.globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto *OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        if (auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get());
            CS && CS->getNumOperands() >= 2)
          if (auto *F = utils::getValueOneLevel<llvm::Function>(CS->getOperand(0)))
            if (auto *AnnotationGL =
                    utils::getValueOneLevel<llvm::GlobalVariable>(CS->getOperand(1)))
              if (auto *Initializer =
                      llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())) {
                llvm::StringRef Annotation = Initializer->getAsCString();
                if (Annotation.compare(SplitterAnnotation) == 0) {
                  SplitterFuncs.insert(F);
                  HIPSYCL_DEBUG_INFO << "Found splitter annotated function " << F->getName()
                                     << "\n";
                } else if (Annotation.compare(KernelAnnotation) == 0) {
                  NDKernels.insert(F);
                  HIPSYCL_DEBUG_INFO << "Found kernel annotated function " << F->getName() << "\n";
                }
              }
      }
    }
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
