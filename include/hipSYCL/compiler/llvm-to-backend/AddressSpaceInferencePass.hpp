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

#ifndef HIPSYCL_SSCP_ADDRESS_SPACE_INFERENCE_PASS_HPP
#define HIPSYCL_SSCP_ADDRESS_SPACE_INFERENCE_PASS_HPP

#include <llvm/IR/PassManager.h>
#include "Utils.hpp"

namespace hipsycl {
namespace compiler {


enum class AddressSpace {
  // These numbers do not reflect actual address spaces,
  // they are just used as indices to look up the address space
  // number in the address space map.
  Generic = 0,
  Global  = 1,
  Local   = 2,
  Private = 3,
  Constant = 4
};

class AddressSpaceMap {
public:
  AddressSpaceMap() {
    for(unsigned i = 0; i < ASMap.size(); ++i)
      ASMap[i] = i;
  }

  AddressSpaceMap(unsigned GenericAS, unsigned GlobalAS, unsigned LocalAS,
                            unsigned PrivateAS, unsigned ConstantAS) {
    (*this)[AddressSpace::Generic] = GenericAS;
    (*this)[AddressSpace::Global] = GlobalAS;
    (*this)[AddressSpace::Local] = LocalAS;
    (*this)[AddressSpace::Private] = PrivateAS;
    (*this)[AddressSpace::Constant] = ConstantAS;
  }

  unsigned operator[](AddressSpace AS) const {
    return ASMap[static_cast<unsigned>(AS)];
  }

  unsigned& operator[](AddressSpace AS) {
    return ASMap[static_cast<unsigned>(AS)];
  }
private:
  std::array<unsigned, 5> ASMap;
};

void rewriteKernelArgumentAddressSpacesTo(unsigned AddressSpace, llvm::Module &M,
                                          const std::vector<std::string> &KernelNames,
                                          PassHandler &PH);

class AddressSpaceInferencePass : public llvm::PassInfoMixin<AddressSpaceInferencePass> {
public:
  AddressSpaceInferencePass(const AddressSpaceMap& Map);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  AddressSpaceMap ASMap;
};

}
}

#endif

