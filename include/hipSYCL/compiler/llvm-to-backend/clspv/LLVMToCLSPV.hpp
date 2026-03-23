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
#pragma

#include "../LLVMToBackend.hpp"

#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {

class LLVMToCLSPVTranslator : public LLVMToBackendTranslator {
public:
  LLVMToCLSPVTranslator(const std::vector<std::string> &KernelNames);

  virtual ~LLVMToCLSPVTranslator() {}

  bool prepareBackendFlavor(llvm::Module &M) override { return true; }
  bool toBackendFlavor(llvm::Module &M, PassHandler &PH) override;
  bool translateToBackendFormat(llvm::Module &FlavoredModule,
                                std::string &Out) override;

protected:
  bool applyBuildOption(const std::string &Option,
                        const std::string &Value) override;
  bool applyBuildFlag(const std::string &Flag) override;
  bool isKernelAfterFlavoring(llvm::Function &F) override;
  AddressSpaceMap getAddressSpaceMap() const override;
  bool optimizeFlavoredIR(llvm::Module &M, PassHandler &PH) override;
  void migrateKernelProperties(llvm::Function *From,
                               llvm::Function *To) override;

private:
  void applyKernelProperties(llvm::Function *F);
  void removeKernelProperties(llvm::Function *F);

  std::vector<std::string> KernelNames;
  unsigned DynamicLocalMemSize = 0;
  std::string MaxPushConstantSize;
  std::string MaxUniformBufferRange;
};

} // namespace compiler
} // namespace hipsycl
