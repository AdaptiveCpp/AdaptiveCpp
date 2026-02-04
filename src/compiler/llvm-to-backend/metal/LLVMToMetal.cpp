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
#include "hipSYCL/compiler/llvm-to-backend/metal/LLVMToMetal.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>


#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>

#include <llvm/Transforms/Utils/LowerSwitch.h>
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ADCE.h"

#include <memory>
#include <cassert>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "Emitter.hpp"

namespace hipsycl {
namespace compiler {

LLVMToMetalTranslator::LLVMToMetalTranslator(const std::vector<std::string>& KernelNames)
  : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::metal), KernelNames, KernelNames}
  , KernelNames(KernelNames)
  , ActualKernelNames(KernelNames.begin(), KernelNames.end())
{ }

LLVMToMetalTranslator::~LLVMToMetalTranslator() = default;

AddressSpaceMap LLVMToMetalTranslator::getAddressSpaceMap() const
{
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 5;
  ASMap[AddressSpace::Constant] = 4;
  ASMap[AddressSpace::AllocaDefault] = 5;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 4;

  return ASMap;
}

bool LLVMToMetalTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return ActualKernelNames.count(F.getName().str()) > 0;
}

bool LLVMToMetalTranslator::prepareBackendFlavor(llvm::Module& M) {
  return true;
}

bool LLVMToMetalTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  // TODO: check layout
#if LLVM_VERSION_MAJOR > 20
  M.setTargetTriple(llvm::Triple("spir64-unknown-unknown"));
#else
  M.setTargetTriple("spir64-unknown-unknown");
#endif
  M.setDataLayout(
    "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
    "1024-A4-n8:16:32:64");

  AddressSpaceMap ASMap = getAddressSpaceMap();

  AddressSpaceInferencePass ASIPass{ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  llvm::StripDebugInfo(M);

  return true;
}

bool LLVMToMetalTranslator::translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) {
  llvm::PassBuilder PB;
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  /* structureed passes */
  llvm::FunctionPassManager FPM;

  FPM.addPass(llvm::PromotePass());
  FPM.addPass(llvm::LowerSwitchPass());
  FPM.addPass(llvm::LoopSimplifyPass());
  FPM.addPass(llvm::LCSSAPass());
  FPM.addPass(llvm::DCEPass());
  FPM.addPass(llvm::ADCEPass());
  FPM.addPass(llvm::StructurizeCFGPass());
  FPM.addPass(llvm::SimplifyCFGPass());
  llvm::ModulePassManager MPM;
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(FlavoredModule, MAM);
  /* */

  std::unordered_set<std::string> kernelNames(KernelNames.begin(), KernelNames.end());
  MetalEmitter emitter(FlavoredModule, kernelNames);
  bool success = emitter.emit(out);
  if (!success) {
    registerError("LLVMToMetal: MetalEmitter failed: " +
                  emitter.errorMessage().value_or("unknown error"));
    return false;
  }
#ifdef ACPP_PRINT_METAL_CODE
  std::cerr << "Generated Metal code:\n" << out << std::endl;
#endif
  return true;
}

void LLVMToMetalTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  ActualKernelNames.erase(From->getName().str());
  ActualKernelNames.insert(To->getName().str());
}


std::unique_ptr<LLVMToBackendTranslator>
createLLVMToMetalTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToMetalTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl