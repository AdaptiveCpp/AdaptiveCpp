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

  void set_max_push_constant_size(uint32_t pc_size) {
    MaxPushConstantSize = pc_size;
  }

  void set_max_uniform_buffer_range(uint32_t ub_range) {
    MaxUniformBufferRange = ub_range;
  }

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
  uint32_t MaxPushConstantSize = 0;
  uint32_t MaxUniformBufferRange = 0;
};

} // namespace compiler
} // namespace hipsycl
