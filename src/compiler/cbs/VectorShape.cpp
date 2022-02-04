//===- src/compiler/cbs/VectorShape.cpp - (s,a) lattice vector shape --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Includes, formatting
//
//===----------------------------------------------------------------------===//
//

#include <cmath>
#include <iostream>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <sstream>

#include "hipSYCL/compiler/cbs/MathUtils.hpp"
#include "hipSYCL/compiler/cbs/VectorShape.hpp"

using namespace llvm;

namespace {

align_t getAlignment(const Constant *c) {
  assert(c);

  if (isa<BasicBlock>(c) || isa<Function>(c))
    return 1;

  // An undef value is never aligned.
  if (isa<UndefValue>(c))
    return 1;

  if (const ConstantInt *cint = dyn_cast<ConstantInt>(c)) {
    return static_cast<align_t>(std::abs(cint->getSExtValue()));
  }

  // Other than that, only integer vector constants can be aligned.
  if (!c->getType()->isVectorTy())
    return 1;

  // A zero-vector is aligned.
  if (isa<ConstantAggregateZero>(c))
    return 0;

  if (const ConstantDataVector *cdv = dyn_cast<ConstantDataVector>(c)) {
    if (!cdv->getElementType()->isIntegerTy())
      return 1;

    const int intValue = (align_t)cast<ConstantInt>(cdv->getAggregateElement(0U))->getZExtValue();

    return static_cast<align_t>(std::abs(intValue));
  }

  assert(isa<ConstantVector>(c));
  const ConstantVector *cv = cast<ConstantVector>(c);

  if (!cv->getType()->getElementType()->isIntegerTy())
    return 1;

  assert(isa<ConstantInt>(cv->getOperand(0)));
  const ConstantInt *celem = cast<ConstantInt>(cv->getOperand(0));
  const int intValue = (align_t)celem->getZExtValue();

  // The vector is aligned if its first element is aligned
  return static_cast<align_t>(std::abs(intValue));
}

} // namespace

namespace hipsycl::compiler {

// undef shape
VectorShape::VectorShape() : stride(0), hasConstantStride(false), alignment(0), defined(false) {}

VectorShape::VectorShape(align_t _alignment)
    : stride(0), hasConstantStride(false), alignment(_alignment), defined(true) {}

// constant stride constructor
VectorShape::VectorShape(stride_t _stride, align_t _alignment)
    : stride(_stride), hasConstantStride(true), alignment(_alignment), defined(true) {}

VectorShape VectorShape::fromConstant(const Constant *C) {
  return VectorShape::uni(getAlignment(C));
}

align_t VectorShape::getAlignmentGeneral() const {
  if (hasConstantStride) {
    if (stride == 0)
      return alignment;
    else
      return gcd(alignment, (align_t)std::abs(stride));
  } else
    return alignment; // General alignment in case of varying shape
}

bool VectorShape::operator==(const VectorShape &a) const {
  return
      // both undef
      (!defined && !a.defined) ||

      // both are defined shapes
      (defined && a.defined && alignment == a.alignment &&
       (
           // either both shapes are varying (with same alignment)
           (!hasConstantStride && !a.hasConstantStride) ||
           // both shapes are strided with same alignment
           (hasConstantStride && a.hasConstantStride && stride == a.stride)));
}

bool VectorShape::operator!=(const VectorShape &a) const { return !(*this == a); }

bool VectorShape::contains(const VectorShape &b) const {
  return join(*this, b) == *this; // TODO efficient implementation
}

bool VectorShape::morePreciseThan(const VectorShape &a) const {
  if (!a.isDefined())
    return false; // Cannot be more precise than bottom
  if (!isDefined())
    return true; // Bottom is more precise then any defined shape

  if (hasConstantStride && !a.hasConstantStride)
    return true; // strided < varying

  // If both are of the same shape, decide by alignment
  if ((hasConstantStride != a.hasConstantStride)) {
    return false; // varying and strided are not comparable
  } else if (hasConstantStride && stride != a.stride) {
    return false; // stride mismatch
  }

  // it comes down to having a coarser alignment
  return (alignment == 0 && a.alignment > 0) || // @this is the zero shape whereas @a is not
         (a.alignment > 0 &&
          (alignment % a.alignment ==
           0)); // the alignment of @this shape is divisible by the alignment of @a

  // either incompatible alignments or @a is zero shape and @this is not
  return false;
}

VectorShape operator-(const VectorShape &a) {
  if (!a.defined || !a.hasConstantStride)
    return a;
  return VectorShape::strided(-a.stride, a.alignment);
}

VectorShape operator+(const VectorShape &a, const VectorShape &b) {
  if (!a.defined || !b.defined)
    return VectorShape::undef();

  if (!a.hasConstantStride || !b.hasConstantStride)
    return VectorShape::varying(gcd(a.getAlignmentGeneral(), b.getAlignmentGeneral()));

  return VectorShape::strided(a.stride + b.stride, gcd(a.alignment, b.alignment));
}

VectorShape operator-(const VectorShape &a, const VectorShape &b) {
  if (!a.defined || !b.defined)
    return VectorShape::undef();

  if (!a.hasConstantStride || !b.hasConstantStride)
    return VectorShape::varying(gcd(a.getAlignmentGeneral(), b.getAlignmentGeneral()));

  return VectorShape::strided(a.stride - b.stride, gcd(a.alignment, b.alignment));
}

VectorShape operator*(int64_t m, const VectorShape &a) {
  if (!a.isDefined())
    return a;

  if (!a.hasStridedShape())
    return VectorShape::varying(std::abs(m) * a.alignment);

  // FIXME overflow
  return VectorShape::strided(m * a.getStride(), std::abs(m) * a.getAlignmentFirst());
}

VectorShape VectorShape::operator/(int64_t M) const {
  if (!isDefined())
    return *this;

  bool IsCleanAlignDiv = (getAlignmentFirst() % M == 0);

  // Result alignment
  align_t NewA = 1;
  if (IsCleanAlignDiv) {
    NewA = getAlignmentFirst() / M;
  }

  if (isVarying() || isUniform()) {
    return isVarying() ? VectorShape::varying(NewA) : VectorShape::uni(NewA);
  }

  assert(hasStridedShape());
  bool IsCleanDiv = (getStride() % M == 0) && IsCleanAlignDiv;
  if (IsCleanDiv) {
    return VectorShape::strided(getStride() % M, NewA);
  }
  return VectorShape::varying(1);
}

VectorShape VectorShape::join(VectorShape a, VectorShape b) {
  if (!a.isDefined())
    return b;
  if (!b.isDefined())
    return a;

  if (a.hasConstantStride && b.hasConstantStride && a.getStride() == b.getStride()) {
    return strided(a.stride, gcd<>(a.alignment, b.alignment));
  } else {
    return varying(gcd(a.getAlignmentGeneral(), b.getAlignmentGeneral()));
  }
}

std::string VectorShape::str() const {
  if (!isDefined()) {
    return "undef_shape";
  }

  std::stringstream ss;
  if (isVarying()) {
    ss << "varying";
  } else if (isUniform()) {
    ss << "uni";
  } else if (isContiguous()) {
    ss << "cont";
  } else {
    ss << "stride(" << stride << ")";
  }

  if (alignment > 1) {
    ss << ", alignment(" << alignment << ", " << getAlignmentGeneral() << ")";
  }

  return ss.str();
}

VectorShape truncateToTypeSize(const VectorShape &a, unsigned typeSize) {
  if (!a.isDefined())
    return a;

  // FIXME
  // observed in SimpleBarrier2D:
  // %41 = <someInst> : stride(32)
  // %42 = trunc i64 i64 %41 to i32 : uni

  // truncate to uniform rule
  if (typeSize == 1 && a.hasStridedShape() && a.getStride() % 2 == 0) {
    return VectorShape::uni();
  }

  return a;
}

std::string VectorShape::serialize() const {
  std::stringstream ss;
  if (isVarying()) {
    ss << "v";
  } else {
    ss << "l" << getStride();
  }
  if (getAlignmentFirst() > 1) {
    ss << "a" << getAlignmentFirst();
  }
  return ss.str();
}

static int ParseInt(StringRef text, int &nextPos) {
  int sign = 1;
  if (text[nextPos] == 'n') {
    nextPos++;
    sign = -1; // negative stride
  }

  int r = 0;
  while (((size_t)nextPos < text.size()) && ('0' <= text[nextPos] && text[nextPos] <= '9')) {
    int d = text[nextPos++] - '0';
    r = 10 * r + d;
  }
  return r * sign;
}

static int ParseAlignment(StringRef text, int &nextPos) {
  if (((size_t)nextPos) >= text.size())
    return 1;
  if (text[nextPos] != 'a')
    return 1;
  nextPos++;
  return ParseInt(text, nextPos);
}

VectorShape VectorShape::parse(StringRef text, int &nextPos) {
  switch (text[nextPos++]) {
  case 'v': {
    int a = ParseAlignment(text, nextPos);
    return VectorShape::varying(a);
  } break;
  case 'l': {
    int s = ParseInt(text, nextPos);
    int a = ParseAlignment(text, nextPos);
    return VectorShape::strided(s, a);
  }
  default:
    abort();
  }
}

} // namespace hipsycl::compiler
