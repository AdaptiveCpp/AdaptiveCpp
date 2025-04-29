//===- LLVMSPIRVLib.h - Read and write SPIR-V binary ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file LLVMSPIRVLib.h
///
/// This files declares functions and passes for translating between LLVM and
/// SPIR-V.
///
///
//===----------------------------------------------------------------------===//
#ifndef SPIRV_H
#define SPIRV_H

#include "LLVMSPIRVOpts.h"

#include <iostream>
#include <string>

namespace llvm {
// Pass initialization functions need to be declared before inclusion of
// PassSupport.h.
class PassRegistry;
void initializeLLVMToSPIRVLegacyPass(PassRegistry &);
void initializeOCLToSPIRVLegacyPass(PassRegistry &);
void initializeOCLTypeToSPIRVLegacyPass(PassRegistry &);
void initializeSPIRVLowerBoolLegacyPass(PassRegistry &);
void initializeSPIRVLowerConstExprLegacyPass(PassRegistry &);
void initializeSPIRVLowerOCLBlocksLegacyPass(PassRegistry &);
void initializeSPIRVLowerMemmoveLegacyPass(PassRegistry &);
void initializeSPIRVLowerSaddWithOverflowLegacyPass(PassRegistry &);
void initializeSPIRVRegularizeLLVMLegacyPass(PassRegistry &);
void initializeSPIRVToOCL12LegacyPass(PassRegistry &);
void initializeSPIRVToOCL20LegacyPass(PassRegistry &);
void initializePreprocessMetadataLegacyPass(PassRegistry &);
void initializeSPIRVLowerBitCastToNonStandardTypeLegacyPass(PassRegistry &);

class ModulePass;
class FunctionPass;
} // namespace llvm

#include "llvm/IR/Module.h"

namespace SPIRV {

class SPIRVModule;

/// \brief Check if a string contains SPIR-V binary.
bool isSpirvBinary(const std::string &Img);

#ifdef _SPIRV_SUPPORT_TEXT_FMT
/// \brief Convert SPIR-V between binary and internal textual formats.
/// This function is not thread safe and should not be used in multi-thread
/// applications unless guarded by a critical section.
/// \returns true if succeeds.
bool convertSpirv(std::istream &IS, std::ostream &OS, std::string &ErrMsg,
                  bool FromText, bool ToText);

/// \brief Convert SPIR-V between binary and internal text formats.
/// This function is not thread safe and should not be used in multi-thread
/// applications unless guarded by a critical section.
bool convertSpirv(std::string &Input, std::string &Out, std::string &ErrMsg,
                  bool ToText);

/// \brief Check if a string contains SPIR-V in internal text format.
bool isSpirvText(std::string &Img);
#endif

/// \brief Load SPIR-V from istream as a SPIRVModule.
/// \returns null on failure.
std::unique_ptr<SPIRVModule> readSpirvModule(std::istream &IS,
                                             std::string &ErrMsg);

/// \brief Load SPIR-V from istream as a SPIRVModule.
/// \returns null on failure.
std::unique_ptr<SPIRVModule> readSpirvModule(std::istream &IS,
                                             const SPIRV::TranslatorOpts &Opts,
                                             std::string &ErrMsg);

struct SPIRVModuleReport {
  SPIRV::VersionNumber Version;
  uint32_t MemoryModel;
  uint32_t AddrModel;
  std::vector<std::string> Extensions;
  std::vector<std::string> ExtendedInstructionSets;
  std::vector<uint32_t> Capabilities;
};
/// \brief Partially load SPIR-V from the stream and decode only selected
/// instructions that are needed to retrieve general information
/// about the module. If this call fails, readSPIRVModule is
/// expected to fail as well.
/// \returns nullopt on failure.
std::optional<SPIRVModuleReport> getSpirvReport(std::istream &IS);
std::optional<SPIRVModuleReport> getSpirvReport(std::istream &IS, int &ErrCode);

struct SPIRVModuleTextReport {
  std::string Version;
  std::string MemoryModel;
  std::string AddrModel;
  std::vector<std::string> Extensions;
  std::vector<std::string> ExtendedInstructionSets;
  std::vector<std::string> Capabilities;
};
/// \brief Create a human-readable form of the report returned by a call to
/// getSpirvReport by decoding its binary fields.
/// \returns String with the human-readable report.
SPIRVModuleTextReport formatSpirvReport(const SPIRVModuleReport &Report);

/// \brief Returns the message associated with the error code.
/// \returns empty string if no known error code is found.
std::string getErrorMessage(int ErrCode);

} // End namespace SPIRV

namespace llvm {

/// \brief Translate LLVM module to SPIR-V and write to ostream.
/// \returns true if succeeds.
bool writeSpirv(Module *M, std::ostream &OS, std::string &ErrMsg);

/// \brief Load SPIR-V from istream and translate to LLVM module.
/// \returns true if succeeds.
bool readSpirv(LLVMContext &C, std::istream &IS, Module *&M,
               std::string &ErrMsg);

/// \brief Translate LLVM module to SPIR-V and write to ostream.
/// \returns true if succeeds.
bool writeSpirv(Module *M, const SPIRV::TranslatorOpts &Opts, std::ostream &OS,
                std::string &ErrMsg);

/// \brief Load SPIR-V from istream and translate to LLVM module.
/// \returns true if succeeds.
bool readSpirv(LLVMContext &C, const SPIRV::TranslatorOpts &Opts,
               std::istream &IS, Module *&M, std::string &ErrMsg);

/// \brief Partially load SPIR-V from the stream and decode only instructions
/// needed to get information about specialization constants.
/// \returns true if succeeds.
struct SpecConstInfoTy {
  uint32_t ID;
  uint32_t Size;
  std::string Type;
};
bool getSpecConstInfo(std::istream &IS,
                      std::vector<SpecConstInfoTy> &SpecConstInfo);

/// \brief Convert a SPIRVModule into LLVM IR.
/// \returns null on failure.
std::unique_ptr<Module>
convertSpirvToLLVM(LLVMContext &C, SPIRV::SPIRVModule &BM, std::string &ErrMsg);

/// \brief Convert a SPIRVModule into LLVM IR using specified options
/// \returns null on failure.
std::unique_ptr<Module> convertSpirvToLLVM(LLVMContext &C,
                                           SPIRV::SPIRVModule &BM,
                                           const SPIRV::TranslatorOpts &Opts,
                                           std::string &ErrMsg);

/// \brief Regularize LLVM module by removing entities not representable by
/// SPIRV.
bool regularizeLlvmForSpirv(Module *M, std::string &ErrMsg);

bool regularizeLlvmForSpirv(Module *M, std::string &ErrMsg,
                            const SPIRV::TranslatorOpts &Opts);

/// \brief Mangle OpenCL builtin function function name.
/// If any type in ArgTypes is a pointer type, it should be represented as a
/// TypedPointerType instead, to faithfully represent the pointer element types
/// for name mangling.
void mangleOpenClBuiltin(const std::string &UnmangledName,
                         ArrayRef<Type *> ArgTypes, std::string &MangledName);

/// Create a pass for translating LLVM to SPIR-V.
ModulePass *createLLVMToSPIRVLegacy(SPIRV::SPIRVModule *);

/// Create a pass for translating OCL C builtin functions to SPIR-V builtin
/// functions.
ModulePass *createOCLToSPIRVLegacy();

/// Create a pass for adapting OCL types for SPIRV.
ModulePass *createOCLTypeToSPIRVLegacy();

/// Create a pass for lowering cast instructions of i1 type.
ModulePass *createSPIRVLowerBoolLegacy();

/// Create a pass for lowering constant expressions to instructions.
ModulePass *createSPIRVLowerConstExprLegacy();

/// Create a pass for removing function pointers related to OCL 2.0 blocks
ModulePass *createSPIRVLowerOCLBlocksLegacy();

/// Create a pass for lowering llvm.memmove to llvm.memcpys with a temporary
/// variable.
ModulePass *createSPIRVLowerMemmoveLegacy();

/// Create a pass for lowering llvm.sadd.with.overflow
ModulePass *createSPIRVLowerSaddWithOverflowLegacy();

/// Create a pass for regularize LLVM module to be translated to SPIR-V.
ModulePass *createSPIRVRegularizeLLVMLegacy();

/// Create a pass for translating SPIR-V Instructions to desired
/// representation in LLVM IR (OpenCL built-ins, SPIR-V Friendly IR, etc.)
ModulePass *createSPIRVBIsLoweringPass(Module &, SPIRV::BIsRepresentation);

/// Create a pass for translating SPIR-V builtin functions to OCL 1.2 builtin
/// functions.
ModulePass *createSPIRVToOCL12Legacy();

/// Create a pass for translating SPIR-V builtin functions to OCL 2.0 builtin
/// functions.
ModulePass *createSPIRVToOCL20Legacy();

/// Create a pass for translating SPIR 1.2/2.0 metadata to SPIR-V friendly
/// metadata.
ModulePass *createPreprocessMetadataLegacy();

/// Create and return a pass that writes the module to the specified
/// ostream.
ModulePass *createSPIRVWriterPass(std::ostream &Str);

/// Create and return a pass that writes the module to the specified
/// ostream.
ModulePass *createSPIRVWriterPass(std::ostream &Str,
                                  const SPIRV::TranslatorOpts &Opts);

/// Create a pass for removing bitcast instructions to non-standard SPIR-V
/// types
FunctionPass *createSPIRVLowerBitCastToNonStandardTypeLegacy(
    const SPIRV::TranslatorOpts &Opts);

} // namespace llvm

#endif // SPIRV_H
