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
#ifndef HIPSYCL_LLVM_TO_PTX_HPP
#define HIPSYCL_LLVM_TO_PTX_HPP


#include "../LLVMToBackend.hpp"

#include <vector>
#include <string>

namespace hipsycl {
namespace compiler {

class LLVMToPtxTranslator : public LLVMToBackendTranslator{
public:
  LLVMToPtxTranslator(const std::vector<std::string>& KernelNames);

  virtual ~LLVMToPtxTranslator() {}

  virtual bool prepareBackendFlavor(llvm::Module& M) override {return true;}
  virtual bool toBackendFlavor(llvm::Module &M, PassHandler& PH) override;
  virtual bool translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) override;
protected:
  virtual bool applyBuildOption(const std::string &Option, const std::string &Value) override;
  virtual bool applyBuildFlag(const std::string &Flag) override;
  
  virtual bool isKernelAfterFlavoring(llvm::Function& F) override;
  virtual AddressSpaceMap getAddressSpaceMap() const override;

  virtual void migrateKernelProperties(llvm::Function* From, llvm::Function* To) override;
private:
  void applyKernelProperties(llvm::Function* F);

  std::vector<std::string> KernelNames;
  unsigned PtxVersion = 30;
  unsigned PtxTarget = 30;

  int FlushDenormalsToZero = -1;
  int PreciseSqrt = -1;
  int PreciseDiv = -1;
};

}
}

#endif
