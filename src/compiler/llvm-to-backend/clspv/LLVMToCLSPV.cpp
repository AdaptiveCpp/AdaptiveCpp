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
#include "hipSYCL/compiler/llvm-to-backend/clspv/LLVMToCLSPV.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/AddrSpaceCastRemovalPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/ConstantAddrSpacePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/FoldChainedGEPsPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/MemsetLoweringPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/RemoveUnusedIntrinsicsPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

namespace {

static const char *DynamicLocalMemArrayName =
    "__acpp_sscp_spirv_dynamic_local_mem";

bool setDynamicLocalMemoryCapacity(llvm::Module &M, unsigned numBytes) {
  llvm::GlobalVariable *GV = M.getGlobalVariable(DynamicLocalMemArrayName);

  if (!GV) {
    // If non-zero number of bytes are needed, not finding the global variable
    // is an error.
    return numBytes == 0;
  }

  const unsigned AddressSpace = GV->getAddressSpace();
  if (numBytes > 0) {
    unsigned numInts = (numBytes + 4 - 1) / 4;
    llvm::Type *T =
        llvm::ArrayType::get(llvm::Type::getInt32Ty(M.getContext()), numInts);

    // Must initialized to undef to be valid SPIR-V
    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage,
        llvm::UndefValue::get(T), GV->getName() + ".resized", nullptr,
        llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal, AddressSpace);

    NewVar->setAlignment(GV->getAlign());
    llvm::Value *V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());
    GV->replaceAllUsesWith(V);
    GV->eraseFromParent();
  } else {
    // If no bytes of local memory used but Global memory exists, then set
    // global variable to a nullptr.
    llvm::Type *T = llvm::PointerType::get(M.getContext(), AddressSpace);
    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage,
        llvm::Constant::getNullValue(T), GV->getName() + ".null", nullptr,
        llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal, AddressSpace);
    GV->replaceAllUsesWith(NewVar);
    GV->eraseFromParent();
  }
  return true;
}

void assignSPIRCallConvention(llvm::Function *F) {
  if (F->getCallingConv() != llvm::CallingConv::SPIR_FUNC)
    F->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  // All callers must use spir_func calling convention
  for (auto U : F->users()) {
    if (auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
      CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    }
  }
}

void rewriteZeroSizeArrayGEPs(llvm::Module &M) {
  llvm::SmallVector<llvm::GetElementPtrInst *> GEPs;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *GEPInst = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
          auto *SourceTy = GEPInst->getSourceElementType();

          if (GEPInst->getNumIndices() > 0 && SourceTy->isArrayTy()) {
            llvm::Value *FirstIndex = GEPInst->idx_begin()->get();
            int64_t FirstIndexVal = -1;
            if (llvm::ConstantInt *CI =
                    llvm::dyn_cast<llvm::ConstantInt>(FirstIndex)) {
              if (CI->getBitWidth() <= 64) {
                FirstIndexVal = CI->getSExtValue();
              }
            }

            if (FirstIndexVal == 0 && SourceTy->getArrayNumElements() == 0) {
              GEPs.push_back(GEPInst);
            }
          }
        }
      }
    }
  }

  for (auto *GEPInst : GEPs) {
    auto *ReplacementType = llvm::ArrayType::get(
        GEPInst->getSourceElementType()->getArrayElementType(), 1);

    llvm::SmallVector<llvm::Value *> Indices;
    for (llvm::Use &Op : GEPInst->indices()) {
      Indices.push_back(Op.get());
    }

    llvm::Value *PointerOperand = GEPInst->getPointerOperand();
    // Old LLVM needs bitcast
#if LLVM_VERSION_MAJOR < 17
    llvm::Type *BitcastTarget = llvm::PointerType::get(
        ReplacementType, GEPInst->getPointerAddressSpace());
    PointerOperand =
        new llvm::BitCastInst(PointerOperand, BitcastTarget, "", GEPInst);
#endif
    llvm::GetElementPtrInst *NewGEP = llvm::GetElementPtrInst::Create(
        ReplacementType, PointerOperand, Indices, "",
        llvmutils::makeInsertionPoint(GEPInst));
    if (GEPInst->isInBounds())
      NewGEP->setIsInBounds(true);
    GEPInst->replaceAllUsesWith(NewGEP);
    GEPInst->dropAllReferences();
    GEPInst->eraseFromParent();
  }
}

} // namespace

LLVMToCLSPVTranslator::LLVMToCLSPVTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::
                                                   compiler_backend::clspv),
                              KN, KN},
      KernelNames{KN} {}

bool LLVMToCLSPVTranslator::toBackendFlavor(llvm::Module &M, PassHandler &PH) {

#if LLVM_VERSION_MAJOR > 20
  M.setTargetTriple(llvm::Triple("spirv64-unknown-vulkan"));
#else
  M.setTargetTriple("spirv64-unknown-vulkan");
#endif
  M.setDataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                  "256-v512:512-v1024:1024-n8:16:32:64-G1");

  rewriteZeroSizeArrayGEPs(M);

  AddressSpaceMap ASMap = getAddressSpaceMap();
  KernelFunctionParameterRewriter ParamRewriter{
      // ByVal attribute for all aggregates passed in by-value
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
      // Those pointers to by-value data should be in private AS
      ASMap[AddressSpace::Private],
      // Actual pointers should be in global memory
      ASMap[AddressSpace::Global],
      // We don't need to wrap pointer types
      false};

  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  for (auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Setting up kernel " << KernelName
                       << "\n";
    if (auto *F = M.getFunction(KernelName)) {
      applyKernelProperties(F);
    }
  }

  for (auto &F : M) {
    if (F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
      // All functions must be marked as spir_func
      assignSPIRCallConvention(&F);
    }
  }

  std::string BuiltinBitcodeFile = common::filesystem::join_path(
      getBitcodePath(), "libkernel-sscp-clspv-full.bc");

#if LLVM_VERSION_MAJOR > 20
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile, M.getTargetTriple().str(),
                             M.getDataLayoutStr()))
#else
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile, M.getTargetTriple(),
                             M.getDataLayoutStr()))
#endif
    return false;

  HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Configuring kernel for "
                     << DynamicLocalMemSize << " bytes of local memory\n";
  if (!setDynamicLocalMemoryCapacity(M, DynamicLocalMemSize)) {
    HIPSYCL_DEBUG_WARNING << "Could not set dynamic local memory size; this "
                             "could imply that local memory "
                             "requested by the application is not actually "
                             "used inside kernels\n";
  }

  // kernel debugging as future work
  llvm::StripDebugInfo(M);

  // Change the behavior of the 'wchar_size' metadata node from 'error'
  // to 'override'. This is due to discrepancy on Windows where we set
  // this to '2' but clspv links in libclc builtins with metadata set
  // to '4'.
  llvm::NamedMDNode *ModFlags = M.getOrInsertModuleFlagsMetadata();
  for (unsigned i = 0; i < ModFlags->getNumOperands(); ++i) {
    llvm::MDNode *Flag = ModFlags->getOperand(i);
    if (llvm::cast<llvm::MDString>(Flag->getOperand(1))->getString() ==
        "wchar_size") {
      auto &Context = M.getContext();
      llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Context);
      llvm::Metadata *Ops[3] = {
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              Int32Ty, llvm::Module::ModFlagBehavior::Override)),
          llvm::MDString::get(Context, "wchar_size"), Flag->getOperand(2)};
      ModFlags->setOperand(i, llvm::MDNode::get(Context, Ops));
      break;
    }
  }

  return true;
}

bool LLVMToCLSPVTranslator::translateToBackendFormat(
    llvm::Module &FlavoredModule, std::string &out) {
  bool Create = true;
  auto consumeError = [&](std::error_code EC) {
    if (EC) {
      if (Create)
        this->registerError("LLVMToCLSPV: Could not create temp file: " +
                            EC.message());
      else
        this->registerError("LLVMToCLSPV: Could not remove temp file: " +
                            EC.message());
      return false;
    }
    return true;
  };

  int InputFD = 0;
  llvm::SmallVector<char> InputFileNameBuf;
  // don't use fs::TempFile, as we can't unlock the file for the clang
  // invocation later... (Windows)
  if (!consumeError(llvm::sys::fs::createTemporaryFile(
          "acpp-sscp-clspv", "bc", InputFD, InputFileNameBuf,
          llvm::sys::fs::OF_None)))
    return false;
  std::string InputFileName = InputFileNameBuf.data();

  llvm::SmallVector<char> OutputFile;
  if (!consumeError(llvm::sys::fs::createTemporaryFile(
          "acpp-sscp-clspv", "spv", OutputFile, llvm::sys::fs::OF_None)))
    return false;
  std::string OutputFileName = OutputFile.data();

  Create = false;
  AtScopeExit DestroyInputFile(
      [&]() { consumeError(llvm::sys::fs::remove(InputFileName)); });
  AtScopeExit DestroyOutputFile(
      [&]() { consumeError(llvm::sys::fs::remove(OutputFileName)); });

  {
    llvm::raw_fd_ostream InputStream{InputFD, true};

    llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
    InputStream.flush();
  }

  std::string CLSPV = HIPSYCL_CLSPV_PATH;

  llvm::SmallVector<std::string> Args{"-x=ir",
                                      "--physical-storage-buffers",
                                      "-spv-version=1.5",
                                      "--arch=spir64",
                                      "-long-vector",
                                      "-o=" + OutputFileName};

  // Workaround for the Qualcomm Adreno driver on Android that struggles to
  // consume SPIR-V where the physical pointer arguments are passed in a push
  // constant with non zero member index. To avoid this pass each argument as
  // its own uniform buffer.
  if (DeviceName.find("Adreno") != std::string::npos) {
    Args.push_back("--pod-ubo");
    Args.push_back("--cluster-pod-kernel-args=0");
  }

  if (!MaxPushConstantSize.empty()) {
    Args.push_back("-max-pushconstant-size=" + MaxPushConstantSize);
  }

  if (!MaxUniformBufferRange.empty()) {
    Args.push_back("-max-ubo-size=" + MaxUniformBufferRange);
  }

  llvm::SmallVector<llvm::StringRef, 16> Invocation{CLSPV};
  for (const auto &A : Args)
    Invocation.push_back(A);
  Invocation.push_back(InputFileName);

  std::string ArgString;
  for (const auto &S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Invoking " << ArgString << "\n";

  int R = executeAndWait(CLSPV, Invocation);

  if (R != 0) {
    this->registerError("LLVMToCLSPV: clspv invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }

  auto ReadResult = llvm::MemoryBuffer::getFile(OutputFileName);

  if (auto Err = ReadResult.getError()) {
    this->registerError("LLVMToCLSPV: Could not read result file" +
                        Err.message());
    return false;
  }

  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToCLSPVTranslator::applyBuildOption(const std::string &Option,
                                             const std::string &Value) {
  if (Option == "spirv-dynamic-local-mem-allocation-size") {
    this->DynamicLocalMemSize = static_cast<unsigned>(std::stoi(Value));
    return true;
  }

  if (Option == "-max-pushconstant-size") {
    this->MaxPushConstantSize = Value;
    return true;
  }

  if (Option == "-max-ubo-size") {
    this->MaxUniformBufferRange = Value;
  }

  if (Option == "-device-name") {
    this->DeviceName = Value;
  }

  return false;
}

bool LLVMToCLSPVTranslator::applyBuildFlag(const std::string &Flag) {
  return false;
}

bool LLVMToCLSPVTranslator::isKernelAfterFlavoring(llvm::Function &F) {
  return F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}

AddressSpaceMap LLVMToCLSPVTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Private] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Constant] = 2;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Generic] = 4;
  ASMap[AddressSpace::AllocaDefault] = 4;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 1;

  return ASMap;
}

bool LLVMToCLSPVTranslator::optimizeFlavoredIR(llvm::Module &M,
                                               PassHandler &PH) {
  bool Result = LLVMToBackendTranslator::optimizeFlavoredIR(M, PH);
  if (!Result)
    return false;

  // Transformation passes to make the LLVM-IR consumable by clspv
  llvm::ModulePassManager MPM;
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(RemoveUnusedIntrinsicsPass()));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(MemsetLoweringPass()));
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(FoldChainedGEPsPass()));
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(AddrSpaceCastRemovalPass()));
  MPM.addPass(ConstantAddrSpacePass());
  MPM.run(M, *PH.ModuleAnalysisManager);

  return Result;
}

void LLVMToCLSPVTranslator::migrateKernelProperties(llvm::Function *From,
                                                    llvm::Function *To) {
  removeKernelProperties(From);
  applyKernelProperties(To);
}

void LLVMToCLSPVTranslator::applyKernelProperties(llvm::Function *F) {
  F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

  llvm::Module &M = *F->getParent();

  if (KnownGroupSizeX != 0 && KnownGroupSizeY != 0 && KnownGroupSizeZ != 0) {
    llvm::SmallVector<llvm::Metadata *> MDs;
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeX)));
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeY)));
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeZ)));

    static const char *ReqdWGSize = "reqd_work_group_size";
    F->setMetadata(ReqdWGSize, llvm::MDNode::get(M.getContext(), MDs));
  }
}

void LLVMToCLSPVTranslator::removeKernelProperties(llvm::Function *F) {
  assignSPIRCallConvention(F);
  for (int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if (F->getArg(i)->hasAttribute(llvm::Attribute::ByVal)) {
      F->getArg(i)->removeAttr(llvm::Attribute::ByVal);
    }
  }
  F->clearMetadata();
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToCLSPVTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToCLSPVTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl
