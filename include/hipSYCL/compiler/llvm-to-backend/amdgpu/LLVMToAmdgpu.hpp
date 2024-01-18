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

#ifndef HIPSYCL_LLVM_TO_AMDGPU_HPP
#define HIPSYCL_LLVM_TO_AMDGPU_HPP


#include "../LLVMToBackend.hpp"

#include <vector>
#include <string>

namespace hipsycl {
namespace compiler {

class LLVMToAmdgpuTranslator : public LLVMToBackendTranslator{
public:
  LLVMToAmdgpuTranslator(const std::vector<std::string>& KernelNames);

  virtual ~LLVMToAmdgpuTranslator() {}

  virtual bool prepareBackendFlavor(llvm::Module& M) override {return true;}
  virtual bool toBackendFlavor(llvm::Module &M, PassHandler& PH) override;
  virtual bool translateToBackendFormat(llvm::Module &FlavoredModule, std::string &Out) override;
protected:
  virtual bool applyBuildOption(const std::string &Option, const std::string &Value) override;
  virtual bool applyBuildFlag(const std::string& Flag) override;
  virtual bool isKernelAfterFlavoring(llvm::Function& F) override;
  virtual AddressSpaceMap getAddressSpaceMap() const override;
private:
  std::vector<std::string> KernelNames;
  std::string RocmDeviceLibsPath;
  std::string RocmPath = ACPP_ROCM_PATH;
  std::string TargetDevice = "gfx900";

  bool hiprtcJitLink(const std::string& Bitcode, std::string& Output);
  bool clangJitLink(llvm::Module& FlavoredModule, std::string& Output);
};

}
}

#endif
