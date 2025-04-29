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

