
#include "hipSYCL/compiler/sscp/KernelOutliningAnalysisPass.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <iostream>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace hipsycl {
namespace compiler {

llvm::AnalysisKey hipsycl::compiler::KernelOutliningAnalysis::Key;

llvm::PreservedAnalyses
KernelOutliningAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* SSCPKernelMarker = "hipsycl_sscp_kernel";
  llvm::SmallPtrSet<llvm::Function*, 16> SSCPKernels;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F && (Annotation.compare(SSCPKernelMarker) == 0)) {
      HIPSYCL_DEBUG_INFO << "Found SSCP Kernel entrypoint: " << F->getName() << "\n";
      SSCPKernels.insert(F);
    }
  });

  return llvm::PreservedAnalyses::all();
}

}
}
