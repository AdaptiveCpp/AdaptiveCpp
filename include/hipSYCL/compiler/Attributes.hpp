/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_ATTRIBUTES_HPP
#define HIPSYCL_ATTRIBUTES_HPP

#include "clang/Sema/Sema.h"

#include <string>

namespace hipsycl {
namespace compiler {

class AddonAttribute
{
  std::string Name;
public:
  AddonAttribute(const std::string& name)
  : Name(name)
  {}

  std::string getString() const
  { return "__attribute__((diagnose_if(false,\""+ Name +",\"warning\")))"; }

  bool describedBy(clang::Attr* attrib) const
  {
    if(clang::isa<clang::DiagnoseIfAttr>(attrib))
    {
      clang::DiagnoseIfAttr* attr = clang::cast<clang::DiagnoseIfAttr>(attrib);
      if(attr->getMessage() == Name)
        return true;
    }
    return false;
  }

  bool isAttachedTo(clang::FunctionDecl *F) const {
    if (clang::Attr *A = F->getAttr<clang::DiagnoseIfAttr>())
      return describedBy(A);
    return false;
  }
};

class KernelAttribute : public AddonAttribute
{
public:
  KernelAttribute()
  : AddonAttribute{"hipsycl_kernel"}
  {}
};

class CustomAttributes
{
public:
  static const KernelAttribute SyclKernel;
};

const KernelAttribute CustomAttributes::SyclKernel = KernelAttribute{};

}
}

#endif
