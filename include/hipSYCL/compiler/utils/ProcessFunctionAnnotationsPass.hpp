/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_PROCESS_FUNCTION_ANNOTATION_PASS_HPP
#define HIPSYCL_PROCESS_FUNCTION_ANNOTATION_PASS_HPP


#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>

#include <vector>

namespace hipsycl {
namespace compiler {
namespace utils {

class ProcessFunctionAnnotationPass : public llvm::PassInfoMixin<ProcessFunctionAnnotationPass> {
public:
  
  ProcessFunctionAnnotationPass(const std::vector<std::string>& AnnotationsToProcess)
  : AnnotationsToProcess{AnnotationsToProcess} {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
    for(const auto& Annotation : AnnotationsToProcess) {
      std::string BuiltinName = "__acpp_function_annotation_"+Annotation;
      processAnnotation(M, Annotation, BuiltinName);
    }
    return llvm::PreservedAnalyses::none();
  }

  const auto& getFoundAnnotations() const {
    return FoundAnnotations;
  }
private:
  void processAnnotation(llvm::Module &M, const std::string &Annotation,
                         const std::string &BuiltinName) {
    
    llvm::SmallVector<llvm::CallBase*> CallsToErase;
    llvm::SmallVector<llvm::Function*> FunctionsToErase;

    for (auto &F : M) {
      if (F.getName().contains(BuiltinName)) {
        for (auto *U : F.users()) {
          if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
            if (auto *ParentBB = CB->getParent()) {
              if (auto *ParentF = ParentBB->getParent()) {
                FoundAnnotations[Annotation].insert(ParentF);
                CallsToErase.push_back(CB);
              }
            }
          }
        }
        FunctionsToErase.push_back(&F);
      }
    }

    for (auto *C : CallsToErase)
      C->eraseFromParent();

    for (auto *F : FunctionsToErase) {
      F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
      F->eraseFromParent();
    }
  }

  std::vector<std::string> AnnotationsToProcess;
  std::unordered_map<std::string, llvm::SmallPtrSet<llvm::Function*, 16>> FoundAnnotations;
};

}
}
}

#endif
