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
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Support/CommandLine.h>

#include <memory>
#include <string>
#include <fstream>
#include <random>


namespace hipsycl {
namespace compiler {

static llvm::cl::opt<bool> SSCPEmitHcf{
    "hipsycl-sscp-emit-hcf", llvm::cl::init(false),
    llvm::cl::desc{"Emit HCF from hipSYCL LLVM SSCP compilation flow"}};

static llvm::cl::opt<bool> PreoptimizeSSCPKernels{
    "hipsycl-sscp-preoptimize", llvm::cl::init(false),
    llvm::cl::desc{
        "Preoptimize SYCL kernels in LLVM IR instead of embedding unoptimized kernels and relying "
        "on optimization at runtime. This is mainly for hipSYCL developers and NOT supported!"}};

static const char *SscpIsHostIdentifier = "__hipsycl_sscp_is_host";
static const char *SscpIsDeviceIdentifier = "__hipsycl_sscp_is_device";
static const char *SscpHcfObjectIdIdentifier = "__hipsycl_local_sscp_hcf_object_id";
static const char *SscpHcfObjectSizeIdentifier = "__hipsycl_local_sscp_hcf_object_size";
static const char *SscpHcfContentIdentifier = "__hipsycl_local_sscp_hcf_content";

template<class IntT>
IntT generateRandomNumber() {
  static std::mutex M;
  static std::random_device Dev;
  static std::mt19937 Rng{Dev()};
  static std::uniform_int_distribution<IntT> dist;

  std::lock_guard<std::mutex> lock {M};
  return dist(Rng);
}

std::unique_ptr<llvm::Module> generateDeviceIR(llvm::Module &M,
                                               std::vector<std::string> &KernelNamesOutput) {

  std::unique_ptr<llvm::Module> DeviceModule = llvm::CloneModule(M);
  DeviceModule->setModuleIdentifier("device." + DeviceModule->getModuleIdentifier());
  
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager DeviceMAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(DeviceMAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, DeviceMAM);

  // Still need to make sure that at least dummy values are there on
  // the device side to avoid undefined references.
  // SscpIsHostIdentifier can also be used in device code.
  S1IRConstantReplacer DeviceSideReplacer{
      {{SscpIsHostIdentifier, 0}, {SscpIsDeviceIdentifier, 1}},
      {{SscpHcfContentIdentifier, 0 /* Dummy value */}, {SscpHcfObjectSizeIdentifier, 0}},
      {{SscpHcfContentIdentifier, std::string{}}}};
  DeviceSideReplacer.run(*DeviceModule, DeviceMAM);

  IRConstant::optimizeCodeAfterConstantModification(*DeviceModule, DeviceMAM);

  if(!PreoptimizeSSCPKernels) {
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O0);
    MPM.run(*DeviceModule, DeviceMAM);
  } else {
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(*DeviceModule, DeviceMAM);
  }

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

    if(SSCPEmitHcf) {
      std::string Filename = M.getSourceFileName()+".hcf";
      std::ofstream OutputFile{Filename.c_str(), std::ios::trunc|std::ios::binary};
      OutputFile.write(HcfString.c_str(), HcfString.size());
      OutputFile.close();
    }
  }

  S1IRConstantReplacer HostSideReplacer{
      {{SscpIsHostIdentifier, 1}, {SscpIsDeviceIdentifier, 0}},
      {{SscpHcfObjectIdIdentifier, HcfObjectId}, {SscpHcfObjectSizeIdentifier, HcfString.size()}},
      {{SscpHcfContentIdentifier, HcfString}}};

  HostSideReplacer.run(M, MAM);
  
  IRConstant::optimizeCodeAfterConstantModification(M, MAM);

  return llvm::PreservedAnalyses::none();
}
}
}
