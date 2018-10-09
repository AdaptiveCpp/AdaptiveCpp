/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#include <string>

#include "clang/AST/Attr.h"

namespace hipsycl {
namespace transform {


class KernelAttribute
{
public:
  static std::string getString()
  { return "__attribute__((section(\"kernel\")))"; }

  static bool describedBy(clang::Attr* attrib)
  {
    if(clang::isa<clang::SectionAttr>(attrib))
    {
      clang::SectionAttr* sectionAttrib = clang::cast<clang::SectionAttr>(attrib);
      if(sectionAttrib->getName().str() == "kernel")
        return true;
    }
    return false;
  }
};


class DeviceAttribute
{
public:
  static std::string getString()
  { return "__attribute__((section(\"device\")))"; }

  static bool describedBy(clang::Attr* attrib)
  {
    if(clang::isa<clang::SectionAttr>(attrib))
    {
      clang::SectionAttr* sectionAttrib = clang::cast<clang::SectionAttr>(attrib);
      if(sectionAttrib->getName().str() == "device")
        return true;
    }
    return false;
  }
};

class HostAttribute
{
public:
  static std::string getString()
  { return "__attribute__((visibility(\"default\")))"; }

  static bool describedBy(clang::Attr* attrib)
  {
    if(clang::isa<clang::VisibilityAttr>(attrib))
    {
      clang::VisibilityAttr* vis = clang::cast<clang::VisibilityAttr>(attrib);
      if(vis->getVisibility() == clang::VisibilityAttr::Default)
        return true;
    }
    return false;
  }
};

}
}

#endif
