//
// Created by joachim on 21.06.21.
//

#ifndef HIPSYCL_BARRIERTAILREPLICATION_HPP
#define HIPSYCL_BARRIERTAILREPLICATION_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace hipsycl {
namespace compiler {

class BarrierTailReplicationPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit BarrierTailReplicationPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL barrier tail replication pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class BarrierTailReplicationPass : public llvm::PassInfoMixin<BarrierTailReplicationPass> {
public:
  explicit BarrierTailReplicationPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_BARRIERTAILREPLICATION_HPP
