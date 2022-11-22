
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Constants.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
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
EntrypointPreparationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  
  static constexpr const char* SSCPKernelMarker = "hipsycl_sscp_kernel";
  static constexpr const char* SSCPOutliningMarker = "hipsycl_sscp_outlining";

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(SSCPKernelMarker) == 0) {
        HIPSYCL_DEBUG_INFO << "Found SSCP kernel: " << F->getName() << "\n";
        this->KernelNames.push_back(F->getName().str());
      }
      if(Annotation.compare(SSCPOutliningMarker) == 0) {
        HIPSYCL_DEBUG_INFO << "Found SSCP outlining entrypoint: " << F->getName() << "\n";
        // Make kernel have external linkage to avoid having everything optimized away
        F->setLinkage(llvm::GlobalValue::ExternalLinkage);
        
        this->OutliningEntrypoints.push_back(F->getName().str());
      }
    }
  });

  return llvm::PreservedAnalyses::none();
}

KernelOutliningPass::KernelOutliningPass(const std::vector<std::string>& OutliningEPs)
: OutliningEntrypoints{OutliningEPs} {}

llvm::PreservedAnalyses
KernelOutliningPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  // Some backends (e.g. PTX) don't like aliases. We need to replace
  // them early on, because it can get difficult to handle them once
  // we have removed what their aliasees. 
  llvm::SmallVector<llvm::GlobalAlias*, 16> AliasesToRemove;
  for(auto& A : M.getAliasList()) 
    AliasesToRemove.push_back(&A);    
  // Need separate iteration, so that we don't erase stuff from the list
  // we are iterating over.
  for(auto* A : AliasesToRemove) {
    if(A) {
      if(A->getAliasee())
        A->replaceAllUsesWith(A->getAliasee());
      A->eraseFromParent();  
    }
  }

  llvm::SmallPtrSet<llvm::Function*, 16> SSCPEntrypoints;
  for(const auto& EntrypointName : OutliningEntrypoints) {
    llvm::Function* F = M.getFunction(EntrypointName);
    
    if(F) {
      SSCPEntrypoints.insert(F);
    }
  }
  llvm::SmallPtrSet<llvm::Function*, 16> DeviceFunctions;

  llvm::CallGraph CG{M};
  for(auto F: SSCPEntrypoints)
    descendCallGraphAndAdd(F, CG, DeviceFunctions);

  
  llvm::SmallVector<llvm::Function*, 16> PureHostFunctions;
  for(auto& F: M.getFunctionList()) {
    if(!DeviceFunctions.contains(&F))
      PureHostFunctions.push_back(&F);
  }

  for(auto F : PureHostFunctions) {
    if(F) {
      //HIPSYCL_DEBUG_INFO << "SSCP Kernel outlining: Stripping function " << F->getName().str() << "\n";
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
  for(auto& G : UnneededGlobals) {
    G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
    G->eraseFromParent();
  } 

  llvm::GlobalOptPass GO;
  GO.run(M, AM);

  return llvm::PreservedAnalyses::none();
}

}
}
