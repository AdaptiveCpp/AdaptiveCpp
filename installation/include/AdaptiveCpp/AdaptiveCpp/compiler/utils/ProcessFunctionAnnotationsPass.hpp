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

#ifndef ACPP_PROCESS_FUNCTION_ANNOTATION_PASS_HPP
#define ACPP_PROCESS_FUNCTION_ANNOTATION_PASS_HPP


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
