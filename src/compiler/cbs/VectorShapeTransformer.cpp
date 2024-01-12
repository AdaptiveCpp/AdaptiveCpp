//===- src/compiler/cbs/VectorShapeTransformer.cpp - (s,a)-lattice abstract transformers --*- C++
//-*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Remove some expensive / unstable actions
// Internalize some functions originally in separate file.
//
//===----------------------------------------------------------------------===//
//

#include "hipSYCL/compiler/cbs/VectorShapeTransformer.hpp"
#include "hipSYCL/compiler/cbs/VectorShape.hpp"
#include "hipSYCL/compiler/cbs/VectorizationInfo.hpp"

#include "hipSYCL/compiler/cbs/MathUtils.hpp"

#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Operator.h>

using namespace hipsycl::compiler;
using namespace llvm;

hipsycl::compiler::VectorShape GenericTransfer(hipsycl::compiler::VectorShape a) {
  if (!a.isDefined())
    return a;
  return a.isUniform() ? hipsycl::compiler::VectorShape::uni()
                       : hipsycl::compiler::VectorShape::varying();
}

template <class... Shapes>
hipsycl::compiler::VectorShape GenericTransfer(hipsycl::compiler::VectorShape a,
                                               Shapes... nextShapes) {
  if (a.isDefined() && !a.isUniform())
    return a.isDefined() ? hipsycl::compiler::VectorShape::varying() : VectorShape::undef();
  else
    return GenericTransfer(nextShapes...);
}

static bool HasSideEffects(const Function &func) {
  if (func.hasFnAttribute(Attribute::ReadOnly) || func.hasFnAttribute(Attribute::ReadNone)) {
    return false;
  }

  return true;
}

VectorShape VectorShapeTransformer::getObservedShape(const BasicBlock &observerBlock,
                                                     const Value &val) const {
  return vecInfo.getObservedShape(LI, observerBlock, val);
}

static Type *getElementType(Type *Ty) {
  if (auto VecTy = dyn_cast<VectorType>(Ty)) {
    return VecTy->getElementType();
  }
#if HAS_TYPED_PTR
  if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    if IS_OPAQUE (PtrTy)
      return nullptr;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return PtrTy->getPointerElementType();
#pragma GCC diagnostic pop
  }
#endif
  if (auto ArrTy = dyn_cast<ArrayType>(Ty)) {
    return ArrTy->getElementType();
  }
  return nullptr;
}

VectorShape VectorShapeTransformer::computeShape(const Instruction &I,
                                                 SmallValVec &taintedOps) const {
  const auto *Phi = dyn_cast<const PHINode>(&I);
  if (Phi)
    return computeShapeForPHINode(*Phi);

  // Otw, this is an instruction
  auto NewShape = computeIdealShapeForInst(I, taintedOps);
  // TODO factor this into vectorShapeTransform. This does not belong here!
  // shape is non-bottom. Apply general refinement rules.
  if (I.getType()->isPointerTy()) {
    // adjust result type to match alignment
    unsigned minAlignment = I.getPointerAlignment(DL).value();
    NewShape.setAlignment(std::max<unsigned>(minAlignment, NewShape.getAlignmentFirst()));
  } else if (isa<FPMathOperator>(I) && !isa<CallInst>(I)) {
    // allow strided/aligned fp values only in fast math mode
    FastMathFlags flags = I.getFastMathFlags();
    if (!flags.isFast() && (NewShape.isDefined() && !NewShape.isUniform())) {
      NewShape = VectorShape::varying();
    }
  }
  return NewShape;
}

VectorShape VectorShapeTransformer::computeIdealShapeForInst(const Instruction &I,
                                                             SmallValVec &taintedOps) const {
  // always default to the naive transformer (only top or bottom)
  if (I.isBinaryOp())
    return computeShapeForBinaryInst(cast<const BinaryOperator>(I));
  if (I.isCast())
    return computeShapeForCastInst(cast<const CastInst>(I));
  if (isa<PHINode>(I))
    return computeShapeForPHINode(cast<const PHINode>(I));
  if (isa<AtomicRMWInst>(I))
    return computeShapeForAtomicRMWInst(cast<const AtomicRMWInst>(I));
  if (isa<AtomicCmpXchgInst>(I))
    return VectorShape::varying();

  const DataLayout &layout = vecInfo.getDataLayout();
  const BasicBlock &BB = *I.getParent();

  switch (I.getOpcode()) {
  case Instruction::Alloca: {
    // AdaptiveCpp CBS: we moved allocas out of the wi-loop.
    return VectorShape::varying();
  }
  case Instruction::Br: {
    const auto &branch = cast<BranchInst>(I);
    assert(branch.isConditional());
    return getObservedShape(BB, *branch.getCondition());
  }
  case Instruction::Switch: {
    const auto &sw = cast<SwitchInst>(I);
    return getObservedShape(BB, *sw.getCondition());
  }
  case Instruction::ICmp: {
    const Value &op1 = *I.getOperand(0);
    const Value &op2 = *I.getOperand(1);

    // Get a shape for op1 - op2 and see if it compares uniform to a full zero-vector
    VectorShape diffShape = getObservedShape(BB, op1) - getObservedShape(BB, op2);
    if (diffShape.isVarying())
      return diffShape;

    CmpInst::Predicate predicate = cast<CmpInst>(I).getPredicate();
    switch (predicate) {
    case CmpInst::Predicate::ICMP_SGT:
    case CmpInst::Predicate::ICMP_UGT:
    case CmpInst::Predicate::ICMP_SLE:
    case CmpInst::Predicate::ICMP_ULE:
      diffShape = -diffShape; // Negate and handle like LESS/GREATER_EQUAL
      // fallthrough
    case CmpInst::Predicate::ICMP_SLT:
    case CmpInst::Predicate::ICMP_ULT:
    case CmpInst::Predicate::ICMP_SGE:
    case CmpInst::Predicate::ICMP_UGE:
      // These coincide because a >= b is equivalent to !(a < b)
      {
        const int vectorWidth = (int)vecInfo.getVectorWidth();
        const int stride = diffShape.getStride();
        const int alignment = diffShape.getAlignmentFirst();

        if (stride >= 0 && alignment >= stride * vectorWidth)
          return VectorShape::uni();
      }
      break;

    case CmpInst::Predicate::ICMP_EQ:
    case CmpInst::Predicate::ICMP_NE: {
      if (diffShape.getStride() == 0)
        return VectorShape::uni();

    } break;

    default:
      break;
    }

    return VectorShape::varying();
  }

  case Instruction::GetElementPtr: {
    const GetElementPtrInst &gep = cast<GetElementPtrInst>(I);
    const Value *pointer = gep.getPointerOperand();

    VectorShape result = getObservedShape(BB, *pointer);
    Type *subT = gep.getPointerOperandType();

    for (const Value *index : make_range(gep.idx_begin(), gep.idx_end())) {
      if (StructType *Struct = dyn_cast<StructType>(subT)) {
        if (!isa<ConstantInt>(index))
          return VectorShape::varying();

        subT = Struct->getTypeAtIndex(index);

        auto structlayout = layout.getStructLayout(Struct);
        unsigned idxconst = (unsigned)cast<ConstantInt>(index)->getSExtValue();
        unsigned elemoffset = (unsigned)structlayout->getElementOffset(idxconst);

        // Behaves like addition, pointer + offset and offset is uniform, hence the stride stays
        align_t newalignment = gcd(result.getAlignmentFirst(), elemoffset);
        result.setAlignment(newalignment);
      } else {
        // NOTE: If indexShape is varying, this still reasons about alignment
        if IS_OPAQUE (subT)
          subT = gep.getSourceElementType();
        else
          subT = getElementType(subT);
        assert(subT && "Unknown LLVM element type .. IR type system change?");

        const size_t typeSizeInBytes = (size_t)layout.getTypeStoreSize(subT);
        result = result + typeSizeInBytes * getObservedShape(BB, *index);
      }
    }

    return result;
  }

  case Instruction::Call: {
    auto &call = cast<CallInst>(I);
    const auto *calledValue = call.getCalledOperand();
    const Function *callee = dyn_cast<Function>(calledValue);
    if (!callee) {
      return VectorShape::varying(); // calling a non-function
    }

    // memcpy shape is uniform if src ptr shape is uniform
    // TODO re-factor into resolver
    Intrinsic::ID id = callee->getIntrinsicID();
    if (id == Intrinsic::memcpy) {
      auto &mcInst = cast<MemCpyInst>(call);
      auto srcShape = getObservedShape(BB, *mcInst.getSource());
      if (srcShape.isDefined() && !srcShape.isUniform())
        taintedOps.push_back(mcInst.getDest());
      return (!srcShape.isDefined() || srcShape.isUniform()) ? srcShape : VectorShape::varying();
    } else if (id == Intrinsic::memmove) {
      auto &movInst = cast<MemMoveInst>(call);
      auto srcShape = getObservedShape(BB, *movInst.getSource());
      if (srcShape.isDefined() && !srcShape.isUniform())
        taintedOps.push_back(movInst.getDest());
      return (!srcShape.isDefined() || srcShape.isUniform()) ? srcShape : VectorShape::varying();
    }

    //    // If the function is rv_align, use the alignment information
    //    if (IsIntrinsic(call, RVIntrinsic::Align)) {
    //      auto shape = getObservedShape(BB, *I.getOperand(0));
    //      shape.setAlignment(cast<ConstantInt>(I.getOperand(1))->getZExtValue());
    //      return shape;
    //    }

    // collect required argument shapes
    // bail if any shape was undefined
    bool allArgsUniform = true;
    size_t numParams = call.arg_size();
    VectorShapeVec callArgShapes;
    for (size_t i = 0; i < numParams; ++i) {
      auto &op = *call.getArgOperand(i);
      auto argShape = getObservedShape(BB, op);
      allArgsUniform &= argShape.isUniform();
      callArgShapes.push_back(argShape);
      if (!argShape.isDefined())
        return VectorShape::undef();
    }

    // known LLVM intrinsic shapes
    if (allArgsUniform && (id == Intrinsic::lifetime_start || id == Intrinsic::lifetime_end)) {
      return VectorShape::uni();
    }

    // next: query resolver mechanism // TODO account for predicate
    // bool hasVaryingBlockPredicate = false;
    // if (!vecInfo.getVaryingPredicateFlag(BB, hasVaryingBlockPredicate)) {
    //   hasVaryingBlockPredicate = false; // assume uniform block pred unless shown otherwise
    // }

    // todo: do we need this in hipSYCL? should be inlined already anyways
    //    auto resolver = platInfo.getResolver(callee->getName(), *callee->getFunctionType(),
    //    callArgShapes,
    //                                         vecInfo.getVectorWidth(), hasVaryingBlockPredicate);
    //    if (resolver) {
    //      return resolver->requestResultShape();
    //    }

    // safe defaults in case we do not know anything about this function.
    bool isUniformCall = allArgsUniform && !HasSideEffects(*callee);
    if (isUniformCall)
      return VectorShape::uni();

    return VectorShape::varying();
  }

  case Instruction::Load: {
    const Value *pointer = I.getOperand(0);
    return VectorShape::join(VectorShape::uni(), getObservedShape(BB, *pointer));
  }

  case Instruction::Store: {
    auto &storeInst = cast<StoreInst>(I);
    auto valShape = getObservedShape(BB, *storeInst.getValueOperand());
    auto ptrShape = getObservedShape(BB, *storeInst.getPointerOperand());
    if (valShape.isDefined() && !valShape.isUniform())
      taintedOps.push_back(storeInst.getPointerOperand());
    return VectorShape::join(ptrShape,
                             (valShape.greaterThanUniform() ? VectorShape::varying() : valShape));
  }

  case Instruction::Select: {
    const Value &condition = *I.getOperand(0);
    const Value &selection1 = *I.getOperand(1);
    const Value &selection2 = *I.getOperand(2);

    const VectorShape &condShape = getObservedShape(BB, condition);
    const VectorShape &sel1Shape = getObservedShape(BB, selection1);
    const VectorShape &sel2Shape = getObservedShape(BB, selection2);

    if (condShape.greaterThanUniform())
      return VectorShape::varying();

    return VectorShape::join(sel1Shape, sel2Shape);
  }

  case Instruction::Fence: {
    return VectorShape::uni();
  }

    // use the generic transfer
  default:
    break;
  }

  return computeGenericArithmeticTransfer(I);
}

VectorShape VectorShapeTransformer::computeGenericArithmeticTransfer(const Instruction &I) const {
  const auto &BB = *I.getParent();

  assert(I.getNumOperands() > 0 &&
         "can not compute arithmetic transfer for instructions w/o operands");

  // interpret as a cascade of generic binary operators
  VectorShape AccuShape = VectorShape::undef();
  for (unsigned i = 0; i < I.getNumOperands(); ++i) {
    VectorShape ObsOpShape = getObservedShape(BB, *I.getOperand(i));
    AccuShape = GenericTransfer(AccuShape, ObsOpShape);
    // early exit
    if (AccuShape.isVarying())
      return AccuShape;
  }
  return AccuShape;
}

VectorShape VectorShapeTransformer::computeShapeForBinaryInst(const BinaryOperator &I) const {
  Value *op1 = I.getOperand(0);
  Value *op2 = I.getOperand(1);

  const auto &BB = *I.getParent();

  // Assume constants are on the RHS
  if (!isa<Constant>(op2) && I.isCommutative())
    std::swap(op1, op2);

  const VectorShape &shape1 = getObservedShape(BB, *op1);
  const VectorShape &shape2 = getObservedShape(BB, *op2);

  const unsigned alignment1 = shape1.getAlignmentFirst();
  const unsigned alignment2 = shape2.getAlignmentFirst();

  const unsigned generalalignment1 = shape1.getAlignmentGeneral();
  const unsigned generalalignment2 = shape2.getAlignmentGeneral();

  switch (I.getOpcode()) {
  case Instruction::Add:
  case Instruction::FAdd:
    return shape1 + shape2;

  case Instruction::Sub:
  case Instruction::FSub:
    return shape1 - shape2;

  // Integer multiplication with a constant simply multiplies the shape offset
  // if existent with the constant value
  // Alignment constants are multiplied
  case Instruction::Mul: {
    if (shape1.isVarying() || shape2.isVarying())
      return VectorShape::varying(generalalignment1 * generalalignment2);

    if (shape1.isUniform() && shape2.isUniform())
      return VectorShape::uni(alignment1 * alignment2);

    // If the constant is known, compute the new shape directly
    if (const ConstantInt *constantOp = dyn_cast<ConstantInt>(op2)) {
      const int c = (int)constantOp->getSExtValue();
      return c * shape1;
    }

    return VectorShape::varying(generalalignment1 * generalalignment2);
  }

  case Instruction::Or: {
    // In some cases Or-with-constant can be interpreted as an Add-with-constant
    // this holds e.g. for: <4, 6, 8, 10> | 1 = <5, 7, 9, 11>
    if (!isa<ConstantInt>(op2))
      break;

    unsigned orConst = cast<ConstantInt>(op2)->getZExtValue();
    VectorShape otherShape = getObservedShape(BB, *op1);

    if (orConst == 0) {
      // no-op
      return otherShape;
    }

    unsigned laneAlignment = otherShape.getAlignmentGeneral();
    if (laneAlignment <= 1)
      break;

    // all bits above constTopBit are zero
    unsigned constTopBit = static_cast<unsigned>(highest_bit(orConst));

    // there is an overlap between the bits of the constant and a possible lane value
    if (constTopBit >= laneAlignment) {
      break;
    }

    // from this point we know that the Or behaves like an Add
    if (otherShape.hasStridedShape()) {
      // the Or operates like an Add with an uniform value
      auto resAlignment = gcd<unsigned>(orConst, otherShape.getAlignmentFirst());
      return VectorShape::strided(otherShape.getStride(), resAlignment);

    } else {
      return VectorShape::varying(gcd<unsigned>(laneAlignment, orConst));
    }

    break;
  }

  case Instruction::Shl: {
    // FIXME overflow
    // interpret shift by constant as multiplication
    if (auto *shiftCI = dyn_cast<ConstantInt>(op2)) {
      unsigned shiftAmount = (int)shiftCI->getZExtValue();
      if (shiftAmount > 0) {
        int64_t factor = ((int64_t)1) << shiftAmount;
        return factor * shape1;
      }
    }
  } break;

  // TODO Instruction::LShr
  case Instruction::AShr: {
    // Special handling for implicit sign extend `(ashr X (shl X $v))`
    auto OpInst = dyn_cast<Instruction>(op1);
    if (OpInst && OpInst->getOpcode() == Instruction::Shl) {
      const auto *ShlAmount = OpInst->getOperand(1);
      if (ShlAmount == op2) {
        // FIXME (ashr X (shlr X 31) 31) is a one-bit sext.. should this
        // really preserve the shape of `X`?
        return vecInfo.getVectorShape(*OpInst->getOperand(0));
      }
    }

    // FIXME overflow
    // interpret shift by constant as division.
    if (auto *shiftCI = dyn_cast<ConstantInt>(op2)) {
      int shiftAmount = (int)shiftCI->getSExtValue();
      if (shiftAmount > 0) {
        int64_t factor = 1 << shiftAmount;
        return shape1 / factor;
      }
    }
  } break;

  case Instruction::SDiv:
  case Instruction::UDiv: {
    const ConstantInt *constDivisor = dyn_cast<ConstantInt>(op2);
    if (constDivisor) {
      return shape1 / constDivisor->getSExtValue();
    }
  } break;

  default:
    break;
  }

  return GenericTransfer(shape1, shape2);
}

bool returnsVoidPtr(const Instruction &inst) {
  if (!isa<CastInst>(inst))
    return false;
  if (!inst.getType()->isPointerTy())
    return false;
  if IS_OPAQUE (inst.getType())
    return true;

#if HAS_TYPED_PTR // otherwise return true from above holds.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return inst.getType()->getPointerElementType()->isIntegerTy(8);
#pragma GCC diagnostic pop
#endif
}

VectorShape VectorShapeTransformer::computeShapeForCastInst(const CastInst &castI) const {
  const auto &BB = *castI.getParent();
  const Value *castOp = castI.getOperand(0);
  const VectorShape &castOpShape = getObservedShape(BB, *castOp);
  const int castOpStride = castOpShape.getStride();

  const int aligned = !returnsVoidPtr(castI) ? castOpShape.getAlignmentFirst() : 1;

  const DataLayout &layout = vecInfo.getDataLayout();

  if (castOpShape.isVarying())
    return castOpShape;

  switch (castI.getOpcode()) {
  case Instruction::IntToPtr: {
    PointerType *DestType = cast<PointerType>(castI.getDestTy());
    if IS_OPAQUE (DestType)
      return VectorShape::strided(castOpStride, 1);

#if HAS_TYPED_PTR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    Type *DestPointsTo = DestType->getPointerElementType();
#pragma GCC diagnostic pop

    // FIXME: void pointers are char pointers (i8*), but what
    // difference is there between a real i8* and a void pointer?
    if (DestPointsTo->isIntegerTy(8))
      return VectorShape::varying();

    unsigned typeSize = (unsigned)layout.getTypeStoreSize(DestPointsTo);

    if (castOpStride % typeSize != 0)
      return VectorShape::varying();

    return VectorShape::strided(castOpStride / typeSize, 1);
#endif
  }

  case Instruction::PtrToInt: {
    Type *SrcType = castI.getSrcTy();
    if IS_OPAQUE (SrcType)
      return VectorShape::strided(castOpStride, aligned);

#if HAS_TYPED_PTR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    Type *SrcElemType = SrcType->getPointerElementType();
#pragma GCC diagnostic pop

    unsigned typeSize = (unsigned)layout.getTypeStoreSize(SrcElemType);

    return VectorShape::strided(typeSize * castOpStride, aligned);
#endif
  }

    // Truncation reinterprets the stride modulo the target type width
    // i16 to i1: stride(odd) -> consecutive, stride(even) -> uniform
  case Instruction::Trunc: {
    Type *destTy = castI.getDestTy();

    return truncateToTypeSize(castOpShape, (unsigned)layout.getTypeStoreSize(destTy));
  }

  // FIXME: is this correct?
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPExt:
    // NOTE: This consciously ignores large absolute values above 2²⁴
  case Instruction::UIToFP:
  case Instruction::SIToFP: {
    return castOpShape;
  }

  // Rounds towards zero: <-1.5f, -0.5f, 0.5f, 1.5f> -> <-1, 0, 0, 1>
  // "If the cast produces an inexact result, how rounding is performed [...]
  // is undefined."
  case Instruction::FPTrunc:
  case Instruction::FPToSI:
  case Instruction::FPToUI: {
    return VectorShape::join(VectorShape::uni(aligned), castOpShape);
  }

  case Instruction::BitCast: {
    Type *srcType = castI.getSrcTy();
    Type *destType = castI.getDestTy();

    // no floating point value involved: keep shape since int<->ptr are compatible
    if (!srcType->isFloatingPointTy() && !destType->isFloatingPointTy()) {
      return castOpShape;
    }

    // Uniform values stay uniform
    if (castOpShape.isUniform())
      return castOpShape;

    // BC from fp<->int/ptr is incomatible -> default to varying shape
    return VectorShape::varying();
  }

  default:
    return VectorShape::join(VectorShape::uni(aligned), castOpShape);
  }
}

VectorShape VectorShapeTransformer::computeShapeForAtomicRMWInst(const AtomicRMWInst &RMW) const {
  if (RMW.getOperation() == AtomicRMWInst::Add || RMW.getOperation() == AtomicRMWInst::Sub) {
    const Value *op1 = RMW.getPointerOperand();
    const Value *op2 = RMW.getValOperand();

    const auto &BB = *RMW.getParent();

    const VectorShape &shape1 = getObservedShape(BB, *op1);

    const ConstantInt *constantOp = dyn_cast<ConstantInt>(op2);
    if (shape1.isUniform() && constantOp) {
      int c = (int)constantOp->getSExtValue();
      if (RMW.getOperation() == AtomicRMWInst::Sub)
        c *= -1;
      return VectorShape::strided(c);
    }
  }

  return VectorShape::varying();
}

VectorShape VectorShapeTransformer::computeShapeForPHINode(const PHINode &Phi) const {
  // catch join divergence cases
  if (!Phi.hasConstantOrUndefValue() && vecInfo.isJoinDivergent(*Phi.getParent())) {
    return VectorShape::varying(); // TODO preserve alignment
  }

  const auto &BB = *Phi.getParent();
  // An incoming value could be divergent by itself.
  // Otherwise, an incoming value could be uniform within the loop
  // that carries its definition but it may appear divergent
  // from outside the loop. This happens when divergent loop exits
  // drop definitions of that uniform value in different iterations.
  //
  // for (int i = 0; i < n; ++i) { // 'i' is uniform inside the loop
  //   if (i % thread_id == 0) break;    // divergent loop exit
  //
  // int divI = i;                 // divI is divergent
  VectorShape accu = VectorShape::undef();
  for (size_t i = 0; i < Phi.getNumIncomingValues(); ++i) {
    const auto &InVal = *Phi.getIncomingValue(i);
    const auto InShape = getObservedShape(BB, InVal);
    accu = VectorShape::join(accu, InShape);
  }

  // joined incoming shapes
  return accu;
}
