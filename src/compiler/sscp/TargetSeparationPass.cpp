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
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/compiler/CompilationState.hpp"
#include "hipSYCL/common/hcf_container.hpp"

#include <cstddef>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

#include <memory>
#include <string>
#include <system_error>
#include <random>

namespace hipsycl {
namespace compiler {

static const char *SscpIsHostIdentifier = "__hipsycl_sscp_is_host";
static const char *SscpIsDeviceIdentifier = "__hipsycl_sscp_is_device";
static const char *SscpHcfObjectIdIdentifier = "__hipsycl_local_sscp_hcf_object_id";
static const char *SscpHcfObjectSizeIdentifier = "__hipsycl_local_sscp_hcf_object_size";
static const char *SscpHcfContentIdentifier = "__hipsycl_local_sscp_hcf_content";

template<class IntT>
class random {
public:
  static IntT generate() {

  }

private:
  random() : Rng{Dev()} {

  }

  std::random_device Dev;
  std::mt19937 Rng;
  std::mutex Mutex;
};

template<class IntT>
IntT generateRandomNumber() {
  static std::mutex M;
  static std::random_device Dev;
  static std::mt19937 Rng{Dev()};
  static std::uniform_int_distribution<IntT> dist;

  std::lock_guard<std::mutex> lock {M};
  return dist(Rng);
}

void removeSuperfluousBranches(llvm::Module& M, llvm::ModuleAnalysisManager& MAM) {

  auto PromoteAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass{});
  auto SCCPAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::SCCPPass{});
  auto ADCEAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::ADCEPass{});

  PromoteAdaptor.run(M, MAM);
  SCCPAdaptor.run(M, MAM);
  ADCEAdaptor.run(M, MAM);
}

std::unique_ptr<llvm::Module> generateDeviceIR(llvm::Module &M,
                                               std::vector<std::string> &KernelNamesOutput) {

  std::unique_ptr<llvm::Module> DeviceModule = llvm::CloneModule(M);

  llvm::ModuleAnalysisManager DeviceMAM;
  // Still need to make sure that at least dummy values are there on
  // the device side to avoid undefined references.
  // SscpIsHostIdentifier can also be used in device code.
  IRConstantReplacer DeviceSideReplacer{
      {{SscpIsHostIdentifier, 0}, {SscpIsDeviceIdentifier, 1}},
      {{SscpHcfContentIdentifier, 0 /* Dummy value */}, {SscpHcfObjectSizeIdentifier, 0}},
      {{SscpHcfContentIdentifier, std::string{}}}};

  DeviceSideReplacer.run(*DeviceModule, DeviceMAM);

  //removeSuperfluousBranches(*DeviceModule, DeviceMAM);

  KernelOutliningPass KP;
  KP.run(*DeviceModule, DeviceMAM);
  KernelNamesOutput = KP.getKernelNames();

  return std::move(DeviceModule);
}

std::string generateHCF(llvm::Module& DeviceModule,
                        std::size_t HcfObjectId,
                        const std::vector<std::string>& KernelNames) {

  std::string ModuleContent;
  llvm::raw_string_ostream OutputStream{ModuleContent};
  llvm::WriteBitcodeToFile(DeviceModule, OutputStream);

  // Debug purposes
  std::error_code EC;
  llvm::raw_fd_ostream test_out{"test.bc", EC};
  llvm::WriteBitcodeToFile(DeviceModule, test_out);
  test_out.close();

  common::hcf_container HcfObject;
  HcfObject.root_node()->set("object-id", std::to_string(HcfObjectId));
  HcfObject.root_node()->set("generator", "hipSYCL SSCP");

  auto *DeviceImagesNodes = HcfObject.root_node()->add_subnode("images");
  auto* LLVMIRNode = DeviceImagesNodes->add_subnode("llvm-ir.global");
  HcfObject.attach_binary_content(LLVMIRNode, ModuleContent);

  auto* KernelsNode = HcfObject.root_node()->add_subnode("kernels");
  for(const auto& KernelName : KernelNames) {
    auto* K = KernelsNode->add_subnode(KernelName);

    auto* F = K->add_subnode("format.llvm-ir");
    auto* ModuleProvider = F->add_subnode("variant.global-module");
    ModuleProvider->set("image-provider", "llvm-ir.global");
  }

  return HcfObject.serialize();
}

llvm::PreservedAnalyses TargetSeparationPass::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {

  // TODO If we know that the SSCP compilation flow is the only one using HCF,
  // we could just enumerate the objects instead of generating (hopefully)
  // unique random numbers.
  std::size_t HcfObjectId = generateRandomNumber<std::size_t>();
  std::string HcfString;


  // Only run SSCP kernel extraction in the host pass
  if(!CompilationStateManager::getASTPassState().isDeviceCompilation()) {
    
    std::vector<std::string> KernelNames;
    std::unique_ptr<llvm::Module> DeviceIR = generateDeviceIR(M, KernelNames);

    HcfString = generateHCF(*DeviceIR, HcfObjectId, KernelNames); 
  }

  IRConstantReplacer HostSideReplacer{
      {{SscpIsHostIdentifier, 1}, {SscpIsDeviceIdentifier, 0}},
      {{SscpHcfObjectIdIdentifier, HcfObjectId}, {SscpHcfObjectSizeIdentifier, HcfString.size()}},
      {{SscpHcfContentIdentifier, HcfString}}};

  HostSideReplacer.run(M, MAM);
  
  removeSuperfluousBranches(M, MAM);

  return llvm::PreservedAnalyses::none();
}
}
}
