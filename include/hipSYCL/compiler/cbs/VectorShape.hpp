//===- hipSYCL/compiler/cbs/VectorShape.hpp - (s,a) lattice vector shape --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Namespace & formatting
//
//===----------------------------------------------------------------------===//
//

#ifndef INCLUDE_RV_VECTORSHAPE_H_
#define INCLUDE_RV_VECTORSHAPE_H_

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace llvm {
class Constant;
}

using align_t = unsigned;
using stride_t = int64_t;

namespace hipsycl::compiler {

// describes how the contents of a vector vary with the vectorized dimension
class VectorShape {
  stride_t stride;
  bool hasConstantStride;
  align_t alignment; // NOTE: General alignment if not hasConstantStride, else alignment of first
  bool defined;

  VectorShape(align_t _alignment);                   // varying
  VectorShape(stride_t _stride, align_t _alignment); // strided

public:
  VectorShape(); // undef

  bool isDefined() const { return defined; }
  stride_t getStride() const { return stride; }
  align_t getAlignmentFirst() const { return alignment; }

  // The maximum common alignment for every possible entry (<6, 8, 10, ...> -> 2)
  align_t getAlignmentGeneral() const;

  void setAlignment(align_t newAlignment) { alignment = newAlignment; }
  void setStride(stride_t newStride) {
    hasConstantStride = true;
    stride = newStride;
  }
  void setVarying(align_t newAlignment) {
    hasConstantStride = false;
    alignment = newAlignment;
  }

  bool isVarying() const { return defined && !hasConstantStride; }
  bool hasStridedShape() const { return defined && hasConstantStride; }
  bool isStrided(stride_t ofStride) const { return hasStridedShape() && stride == ofStride; }
  bool isStrided() const { return hasStridedShape() && stride != 0 && stride != 1; }
  bool isUniform() const { return isStrided(0); }
  bool greaterThanUniform() const { return !isUniform() && isDefined(); }
  inline bool isContiguous() const { return isStrided(1); }

  static VectorShape varying(align_t aligned = 1) { return VectorShape(aligned); }
  static VectorShape strided(stride_t stride, align_t aligned = 1) {
    return VectorShape(stride, aligned);
  }
  static inline VectorShape uni(align_t aligned = 1) { return strided(0, aligned); }
  static inline VectorShape cont(align_t aligned = 1) { return strided(1, aligned); }
  static VectorShape undef() { return VectorShape(); } // bot

  static VectorShape fromConstant(const llvm::Constant *C);

  static VectorShape join(VectorShape a, VectorShape b);

  bool operator==(const VectorShape &a) const;
  bool operator!=(const VectorShape &a) const;
  VectorShape operator/(int64_t D) const;

  // lattice order
  bool morePreciseThan(
      const VectorShape &a) const; // whether @this is less than @a according to lattice order
  bool contains(const VectorShape &b) const; // whether join(@this, @b) == @this

  friend VectorShape operator-(const VectorShape &a);
  friend VectorShape operator+(const VectorShape &a, const VectorShape &b);
  friend VectorShape operator-(const VectorShape &a, const VectorShape &b);
  friend VectorShape operator*(int64_t m, const VectorShape &a);
  friend VectorShape truncateToTypeSize(const VectorShape &a, unsigned typeSize);

  std::string str() const;

  static VectorShape truncateToTypeSize(const VectorShape &a, unsigned typeSize);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const VectorShape &shape) {
    return O << shape.str();
  }

  std::string serialize() const;
  // parse the next shape in @text (starting from nextPos) and return the parsed shape
  // (setting @nextPos on the next position after the last used character)
  static VectorShape parse(llvm::StringRef text, int &nextPos);
};

typedef std::vector<VectorShape> VectorShapeVec;
} // namespace hipsycl::compiler

#endif /* INCLUDE_RV_VECTORSHAPE_H_ */