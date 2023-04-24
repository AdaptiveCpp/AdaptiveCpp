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

#ifndef HIPSYCL_SSCP_ADDRESS_SPACE_MAP_HPP
#define HIPSYCL_SSCP_ADDRESS_SPACE_MAP_HPP

#include <array>

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
  Constant = 4,
  AllocaDefault = 5,
  GlobalVariableDefault = 6,
  ConstantGlobalVariableDefault = 7
};

class AddressSpaceMap {
public:
  AddressSpaceMap() {
    for(unsigned i = 0; i < ASMap.size(); ++i)
      ASMap[i] = i;
  }

  AddressSpaceMap(unsigned GenericAS, unsigned GlobalAS, unsigned LocalAS, unsigned PrivateAS,
                  unsigned ConstantAS, unsigned AllocaDefaultAS, unsigned GlobalVariableDefaultAS,
                  unsigned ConstantGlobalVariableDefaultAS) {
    (*this)[AddressSpace::Generic] = GenericAS;
    (*this)[AddressSpace::Global] = GlobalAS;
    (*this)[AddressSpace::Local] = LocalAS;
    (*this)[AddressSpace::Private] = PrivateAS;
    (*this)[AddressSpace::Constant] = ConstantAS;
    (*this)[AddressSpace::AllocaDefault] = AllocaDefaultAS;
    (*this)[AddressSpace::GlobalVariableDefault] = GlobalVariableDefaultAS;
    (*this)[AddressSpace::ConstantGlobalVariableDefault] = ConstantGlobalVariableDefaultAS;
  }

  AddressSpaceMap(unsigned GenericAS, unsigned GlobalAS, unsigned LocalAS, unsigned PrivateAS,
                  unsigned ConstantAS)
      : AddressSpaceMap{GenericAS,  GlobalAS,  LocalAS,  PrivateAS,
                        ConstantAS, PrivateAS, GlobalAS, ConstantAS} {}

  unsigned operator[](AddressSpace AS) const {
    return ASMap[static_cast<unsigned>(AS)];
  }

  unsigned& operator[](AddressSpace AS) {
    return ASMap[static_cast<unsigned>(AS)];
  }
private:
  std::array<unsigned, 8> ASMap;
};

}
}

#endif

