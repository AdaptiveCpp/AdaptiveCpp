
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <iostream>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace hipsycl {
namespace compiler {

template<class FunctionSetT>
void descendCallGraphAndAdd(llvm::Function* F, llvm::CallGraph& CG, FunctionSetT& Set){
  if(!F || Set.contains(F))
    return;
  
  Set.insert(F);
  llvm::CallGraphNode* CGN = CG.getOrInsertFunction(F);
  if(!CGN)
    return;
  for(unsigned i = 0; i < CGN->size(); ++i){
    descendCallGraphAndAdd((*CGN)[i]->getFunction(), CG, Set);
  }
}

llvm::PreservedAnalyses
KernelOutliningPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* SSCPKernelMarker = "hipsycl_sscp_kernel";
  llvm::SmallPtrSet<llvm::Function*, 16> SSCPKernels;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F && (Annotation.compare(SSCPKernelMarker) == 0)) {
      HIPSYCL_DEBUG_INFO << "Found SSCP Kernel entrypoint: " << F->getName() << "\n";
      SSCPKernels.insert(F);
    }
  });

  llvm::SmallPtrSet<llvm::Function*, 16> DeviceFunctions;

  llvm::CallGraph CG{M};
  for(auto F: SSCPKernels)
    descendCallGraphAndAdd(F, CG, DeviceFunctions);

  
  llvm::SmallVector<llvm::Function*, 16> PureHostFunctions;
  for(auto& F: M.getFunctionList()) {
    if(!DeviceFunctions.contains(&F))
      PureHostFunctions.push_back(&F);
  }

  for(auto F : PureHostFunctions) {
    if(F) {
      HIPSYCL_DEBUG_INFO << "SSCP Kernel outlining: Stripping host function " << F->getName().str() << "\n";
      F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
      F->eraseFromParent();
    }
  }

  llvm::SmallVector<llvm::GlobalVariable*, 16> UnneededGlobals;
  for(auto& G: M.getGlobalList()) {
    G.removeDeadConstantUsers();
    if(G.getNumUses() == 0)
      UnneededGlobals.push_back(&G);
  }

  for(auto G : UnneededGlobals) {
    G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
    G->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

}
}
