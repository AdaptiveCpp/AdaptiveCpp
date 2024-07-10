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
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackendTool.hpp"
#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirvFactory.hpp"
#include <memory>

namespace tool = hipsycl::compiler::translation_tool;

std::unique_ptr<hipsycl::compiler::LLVMToBackendTranslator>
createSpirvTranslator(const hipsycl::common::hcf_container& HCF) {
  std::vector<std::string> KernelNames;
  if(!tool::getHcfKernelNames(HCF, KernelNames)) {
    return nullptr;
  }
  return hipsycl::compiler::createLLVMToSpirvTranslator(KernelNames);
}

int main(int argc, char* argv[]) {
  return tool::LLVMToBackendToolMain(argc, argv, createSpirvTranslator);
}
