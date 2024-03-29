/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_LLVM_TO_BACKEND_HPP
#define HIPSYCL_LLVM_TO_BACKEND_HPP


// Note: This file should not include any LLVM headers or include
// dependencies that rely on LLVM headers in order to not spill
// LLVM code into the hipSYCL runtime.
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <typeinfo>
#include <functional>
#include "AddressSpaceMap.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace llvm {
class Module;
class Function;
}

namespace hipsycl {
namespace compiler {

struct PassHandler;

struct TranslationHints {
  std::optional<std::size_t> RequestedLocalMemSize;
  std::optional<std::size_t> SubgroupSize;
  std::optional<rt::range<3>> WorkGroupSize;
};

class LLVMToBackendTranslator {
public:
  LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
    const std::vector<std::string>& OutliningEntrypoints,
    const std::vector<std::string>& KernelNames);

  virtual ~LLVMToBackendTranslator() {}

  // Do not use inside llvm-to-backend infrastructure targets to avoid
  // requiring RTTI-enabled LLVM
  template<auto& ConstantName, class T>
  void setS2IRConstant(const T& value) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "Unsupported type for S2 IR constant");

    std::string name = typeid(__hipsycl_sscp_s2_ir_constant<ConstantName, T>).name();
    setS2IRConstant<T>(name, value);
  }

  template<class T>
  void setS2IRConstant(const std::string& name, T value) {
    setS2IRConstant(name, static_cast<const void*>(&value));
  }

  void setS2IRConstant(const std::string& name, const void* ValueBuffer);
  void specializeKernelArgument(const std::string &KernelName, int ParamIndex,
                                const void *ValueBuffer);

  bool setBuildFlag(const std::string &Flag);
  bool setBuildOption(const std::string &Option, const std::string &Value);
  bool setBuildToolArguments(const std::string &ToolName, const std::vector<std::string> &Args);

  template<class T>
  bool setBuildOption(const std::string& Option, const T& Value) {
    return setBuildOption(Option, std::to_string(Value));
  }

  // Does partial transformation to backend-flavored LLVM IR
  bool partialTransformation(const std::string& LLVMIR, std::string& out);

  // Does full transformation to backend specific format
  bool fullTransformation(const std::string& LLVMIR, std::string& out);
  bool prepareIR(llvm::Module& M);
  bool translatePreparedIR(llvm::Module& FlavoredModule, std::string& out);


  const std::vector<std::string>& getErrorLog() const {
    return Errors;
  }

  // Returns IR that caused the error in case an error occurs
  const std::string& getFailedIR() const {
    return ErroringCode;
  }

  const std::vector<std::string>& getOutliningEntrypoints() const {
    return OutliningEntrypoints;
  }

  const std::vector<std::string>& getKernels () const {
    return Kernels;
  }

  std::string getErrorLogAsString() const {
    std::string Result;
    for(int i = 0; i < getErrorLog().size(); ++i) {
      Result += std::to_string(i);
      Result += ": ";
      Result += getErrorLog()[i] + '\n';
    }
    return Result;
  }

  int getBackendId() const {
    return S2IRConstantBackendId;
  }

  using SymbolListType = std::vector<std::string>;

  class ExternalSymbolResolver {
  public:
    using LLVMModuleId = unsigned long long;
    // SymbolToModuleMapper is responsible for return a list of identifiers of LLVM modules
    // that should be linked in order to resolve the provided symbol list.
    using SymbolsToModuleIdMapperType =
        std::function<std::vector<LLVMModuleId>(const SymbolListType &SymbolList)>;
    // BitcodeStringRetriever will return the IR bitcode string as well as the imported symbols,
    // given a unique LLVM module id.
    using BitcodeStringRetrieverType = std::function<std::string (LLVMModuleId, SymbolListType&)>;

    ExternalSymbolResolver() = default;
    ExternalSymbolResolver(const SymbolsToModuleIdMapperType &SymbolMapper,
                           const BitcodeStringRetrieverType &Retriever,
                           const SymbolListType &ImportedSymbols)
        : SymbolModuleMapper{SymbolMapper}, BitcodeRetriever{Retriever}, ImportedSymbols{
                                                                             ImportedSymbols} {}

    auto mapSymbolsToModuleIds(const SymbolListType& symbols) const {
      return SymbolModuleMapper(symbols);
    }

    auto retrieveBitcode(LLVMModuleId MID, SymbolListType& ImportedSymbolsOut) const {
      return BitcodeRetriever(MID, ImportedSymbolsOut);
    }

    // retrieve imported symbols for the primary bitcode file
    const SymbolListType& getImportedSymbols() const {
      return ImportedSymbols;
    }
  private:
    SymbolsToModuleIdMapperType SymbolModuleMapper;
    BitcodeStringRetrieverType BitcodeRetriever;
    SymbolListType ImportedSymbols;
  };

  void provideExternalSymbolResolver(ExternalSymbolResolver Resolver);

protected:
  virtual AddressSpaceMap getAddressSpaceMap() const = 0;
  virtual bool isKernelAfterFlavoring(llvm::Function& F) = 0;
  virtual bool applyBuildFlag(const std::string &Flag) { return false; }
  virtual bool applyBuildOption(const std::string &Option, const std::string &Value) { return false; }
  virtual bool applyBuildToolArguments(const std::string &ToolName,
                                       const std::vector<std::string> &Args) {
    return false;
  }

  // Link against bitcode contained in file or string. If ForcedTriple/ForcedDataLayout are non-empty,
  // sets triple and data layout in contained bitcode to the provided values.
  
  bool linkBitcodeFile(llvm::Module &M, const std::string &BitcodeFile,
                       const std::string &ForcedTriple = "",
                       const std::string &ForcedDataLayout = "",
                       bool LinkOnlyNeeded = true);
  bool linkBitcodeString(llvm::Module &M, const std::string &Bitcode,
                         const std::string &ForcedTriple = "",
                         const std::string &ForcedDataLayout = "",
                         bool LinkOnlyNeeded = true);
  // If backend needs to set IR constants, it should do so here.
  virtual bool prepareBackendFlavor(llvm::Module& M) = 0;
  // Transform LLVM IR as much as required to backend-specific flavor
  virtual bool toBackendFlavor(llvm::Module &M, PassHandler& PH) = 0;
  virtual bool translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) = 0;

  // By default, just runs regular O3 pipeline. Backends may override
  // if they want to do something more specific.
  virtual bool optimizeFlavoredIR(llvm::Module& M, PassHandler& PH);

  void registerError(const std::string& E) {
    Errors.push_back(E);
  }

  // These will be non-zero if work group sizes are known at jit time.
  // Backends should check these values for being != 0 before using them.
  int KnownGroupSizeX = 0;
  int KnownGroupSizeY = 0;
  int KnownGroupSizeZ = 0;

  // Will be >= 0 if set by option. Backends using this should therefore check >= 0.
  std::int64_t KnownLocalMemSize = -1;

  bool GlobalSizesFitInInt = false;
  bool IsFastMath = false;

private:

  void resolveExternalSymbols(llvm::Module& M);
  void setFailedIR(llvm::Module& M);

  int S2IRConstantBackendId;
  
  std::vector<std::string> OutliningEntrypoints;
  std::vector<std::string> Kernels;

  std::vector<std::string> Errors;
  std::unordered_map<std::string, std::function<void(llvm::Module &)>> SpecializationApplicators;
  ExternalSymbolResolver SymbolResolver;
  bool HasExternalSymbolResolver = false;

  // In case an error occurs, the code will be stored here
  std::string ErroringCode;

};

}
}

#endif
