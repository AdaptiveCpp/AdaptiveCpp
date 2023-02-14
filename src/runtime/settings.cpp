/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/settings.hpp"

namespace hipsycl {
namespace rt {

std::istream &operator>>(std::istream &istr, scheduler_type &out) {
  std::string str;
  istr >> str;
  if (str == "direct")
    out = scheduler_type::direct;
  else if (str == "unbound")
    out = scheduler_type::unbound;
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

std::istream &operator>>(std::istream &istr, std::vector<rt::backend_id> &out) {
  std::string str;
  istr >> str;
  // have to copy, as otherweise might be interpreted as failing, although everything is fine.
  std::istringstream istream{str};

  std::string name;
  while(std::getline(istream, name, ';')) {
    if(name.empty())
      continue;
    std::transform(name.cbegin(), name.cend(), name.begin(), ::tolower);

    if (name == "cuda") {
      out.push_back(rt::backend_id::cuda);
    } else if (name == "hip") {
      out.push_back(rt::backend_id::hip);
    } else if (name == "ze") {
      out.push_back(rt::backend_id::level_zero);
    } else if (name == "omp") {
      // looking for this, even though we have to allow it always.
      out.push_back(rt::backend_id::omp);
    } else {
      istr.setstate(std::ios_base::failbit);
      HIPSYCL_DEBUG_WARNING << "'" << name << "' is not a known backend name." << std::endl;
      break;
    }
  }
  return istr;
}

std::istream &operator>>(std::istream &istr, default_selector_behavior& out) {
  std::string str;
  istr >> str;
  if (str == "strict")
    out = default_selector_behavior::strict;
  else if (str == "multigpu")
    out = default_selector_behavior::multigpu;
  else if (str == "system")
    out = default_selector_behavior::system;
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

}
}