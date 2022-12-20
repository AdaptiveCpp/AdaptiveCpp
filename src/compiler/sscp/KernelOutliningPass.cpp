
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/common/debug.hpp"
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

namespace {

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

// Check whether F is used by an instruction from any function contained in
// a set S
template<class Set>
bool isCalledFromAnyFunctionOfSet(llvm::Function* F, const Set& S) {
  for(auto* U : F->users()) {
    if(auto* I = llvm::dyn_cast<llvm::Instruction>(U)) {
      auto* UsingFunc = I->getFunction();
      if(UsingFunc && S.contains(UsingFunc)) {
        return true;
      }
    }
  }
  return false;
}

}

llvm::PreservedAnalyses
EntrypointPreparationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  
  static constexpr const char* SSCPKernelMarker = "hipsycl_sscp_kernel";
  static constexpr const char* SSCPOutliningMarker = "hipsycl_sscp_outlining";

  llvm::SmallSet<std::string, 16> Kernels;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(SSCPKernelMarker) == 0) {
        HIPSYCL_DEBUG_INFO << "Found SSCP kernel: " << F->getName() << "\n";
        this->KernelNames.push_back(F->getName().str());
        Kernels.insert(F->getName().str());
      }
      if(Annotation.compare(SSCPOutliningMarker) == 0) {
        HIPSYCL_DEBUG_INFO << "Found SSCP outlining entrypoint: " << F->getName() << "\n";
        // Make kernel have external linkage to avoid having everything optimized away
        F->setLinkage(llvm::GlobalValue::ExternalLinkage);
        
        // If we have a definition, we need to perform outlining.
        // Otherwise, we would need to treat the function as imported --
        // however this cannot really happen as clang does not codegen our
        // attribute((annotate("hipsycl_sscp_outlining"))) for declarations
        // without definition.
        if(F->getBasicBlockList().size() > 0)
          this->OutliningEntrypoints.push_back(F->getName().str());
      }
    }
  });


  for(const auto& EP : OutliningEntrypoints) {
    if(!Kernels.contains(EP)) {
      NonKernelOutliningEntrypoints.push_back(EP);
    }
  }

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

  for(auto* F : DeviceFunctions) {
    //HIPSYCL_DEBUG_INFO << "SSCP Kernel outlining: Function is device function: "
    //                   << F->getName().str() << "\n";
  }
  
  llvm::SmallVector<llvm::Function*, 16> PureHostFunctions;
  for(auto& F: M.getFunctionList()) {
    // Called Intrinsics don't show up in our device functions list,
    // so we need to treat them specially
    if(F.isIntrinsic()) {
      if(!isCalledFromAnyFunctionOfSet(&F, DeviceFunctions)) {
        PureHostFunctions.push_back(&F);
      }
    } else if(!DeviceFunctions.contains(&F)) {
      PureHostFunctions.push_back(&F);
    }
  }

  for(auto F : PureHostFunctions) {
    if(F) {
      bool SafeToRemove = !isCalledFromAnyFunctionOfSet(F, DeviceFunctions);
      if(!SafeToRemove) {
        HIPSYCL_DEBUG_WARNING << "KernelOutliningPass: Attempted to remove " << F->getName()
                              << ", but it is still used by functions marked as device functions.\n";
      }
      // Better safe than sorry!
      if(SafeToRemove) {
        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->eraseFromParent();
      }
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
