/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/sscp/TargetSeparationPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/SCCP.h>

namespace hipsycl {
namespace compiler {

static std::size_t __hipsycl_local_sscp_hcf_object_id;
static std::size_t __hipsycl_local_sscp_hcf_object_size;
static const char* __hipsycl_local_sscp_hcf_content;
static int  __hipsycl_sscp_is_host;

void removeSuperfluousBranches(llvm::Module& M, llvm::ModuleAnalysisManager& MAM) {

  for(auto& F: M.getFunctionList()) {
    llvm::FunctionAnalysisManager FM;
    llvm::DCEPass dce;
    llvm::SCCPPass sccp;

    sccp.run(F, FM);
    dce.run(F, FM);
  }
}

std::string generateHCFKernels(llvm::Module& M,
                              llvm::ModuleAnalysisManager& MAM,
                              std::size_t HcfObjectId) {
  IRConstantReplacer DeviceSideReplacer {
    {{"__hipsycl_sscp_is_host", 0}},
    {{"__hipsycl_local_sscp_hcf_object_id", 0 /* Dummy value */},
    {"__hipsycl_local_sscp_hcf_object_size", 0}},
     {{"__hipsycl_local_sscp_hcf_content", std::string{}}}
  };

  DeviceSideReplacer.run(M, MAM);

  //removeSuperfluousBranches(M, MAM);

  return "This is an HCF string";
}

llvm::PreservedAnalyses TargetSeparationPass::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {

  std::unique_ptr<llvm::Module> DeviceModule = llvm::CloneModule(M);

  std::size_t HcfObjectId = 0; // TODO
  std::string HcfString = generateHCFKernels(*DeviceModule, MAM, HcfObjectId); 

  IRConstantReplacer HostSideReplacer {
    {{"__hipsycl_sscp_is_host", 1}},
    {{"__hipsycl_local_sscp_hcf_object_id", HcfObjectId},
    {"__hipsycl_local_sscp_hcf_object_size", HcfString.size()}},
     {{"__hipsycl_local_sscp_hcf_content", HcfString}}
  };

  HostSideReplacer.run(M, MAM);

  removeSuperfluousBranches(M, MAM);

  return llvm::PreservedAnalyses::none();
}
}
}
