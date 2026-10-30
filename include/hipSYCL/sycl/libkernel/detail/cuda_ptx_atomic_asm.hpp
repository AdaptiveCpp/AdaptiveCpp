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
#ifndef HIPSYCL_NVIDIA_PTX_ATOMIC_PRIMITIVES_HPP
#define HIPSYCL_NVIDIA_PTX_ATOMIC_PRIMITIVES_HPP
#include "hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp"

namespace PTXAtomicDetail 
{
  using OrderSSCP = __acpp_sscp_memory_order;
  using ScopeSSCP  = __acpp_sscp_memory_scope;

  // Reports violated backend preconditions through AdaptiveCpp's device
  // assertion builtin when NDEBUG is not defined.
  inline void debug_assert(
      bool condition,
      const char* message,
      const char* file,
      __acpp_uint32 line,
      const char* function)
  {
  #ifndef NDEBUG
      if (!condition) {__acpp_sscp_assert_fail(message, file, line, function);}
  #else
      (void)condition; (void)message; (void)file; (void)line; (void)function;
  #endif
  }

  // Emits a sequentially consistent PTX fence at the requested scope.
  // System scope uses .sys, device scope uses .gpu, and the smaller
  // SYCL scopes use the conservative work-group-wide .cta scope.
  // The fence participates in the PTX ABI lowering of SC atomic operations.
  inline void fence_seq_cst(ScopeSSCP scope)
  {
    if (scope == ScopeSSCP::system) 
    {
      asm volatile("fence.sc.sys;" ::: "memory");
    }
    else if (scope == ScopeSSCP::device) 
    {
      asm volatile("fence.sc.gpu;" ::: "memory");
    }
    else 
    {
      asm volatile("fence.sc.cta;" ::: "memory");
    }
  }


  // Generates 32-bit and 64-bit PTX atomic-load helpers.
  // SEM selects relaxed or acquire ordering; BITS selects the operand width;
  // REG selects the corresponding PTX register constraint.
  // The requested memory scope selects .cta, .gpu, or .sys.
  // The memory clobber constrains compiler reordering around the inline asm.
  #define ACPP_PTX_LOAD(NAME, SEM, TYPE, BITS, REG)           \
  inline TYPE NAME(TYPE* ptr, ScopeSSCP scope)                \
  {                                                           \
    TYPE result;                                              \
    if (scope == ScopeSSCP::system)                           \
    {                                                         \
      asm volatile(                                           \
        "ld." SEM ".sys.u" BITS " %0, [%1];" :                \
        "=" REG (result) :                                    \
        "l" (ptr) :                                           \
        "memory"                                              \
      );                                                      \
    }                                                         \
    else if (scope == ScopeSSCP::device)                      \
    {                                                         \
      asm volatile(                                           \
        "ld." SEM ".gpu.u" BITS " %0, [%1];" :                \
        "=" REG (result) :                                    \
        "l" (ptr) :                                           \
        "memory"                                              \
      );                                                      \
    }                                                         \
    else                                                      \
    {                                                         \
      asm volatile(                                           \
        "ld." SEM ".cta.u" BITS " %0, [%1];" :                \
        "=" REG (result) :                                    \
        "l" (ptr) :                                           \
        "memory"                                              \
      );                                                      \
    }                                                         \
    return result;                                            \
  }
  ACPP_PTX_LOAD(relaxed_load, "relaxed", __acpp_int32, "32", "r")
  ACPP_PTX_LOAD(relaxed_load, "relaxed", __acpp_int64, "64", "l")
  ACPP_PTX_LOAD(acquire_load, "acquire", __acpp_int32, "32", "r")
  ACPP_PTX_LOAD(acquire_load, "acquire", __acpp_int64, "64", "l")
  #undef ACPP_PTX_LOAD

  // Generates 32-bit and 64-bit PTX atomic-store helpers.
  // SEM selects relaxed or release ordering, while the requested scope
  // selects the PTX synchronization domain.
  // The input register contains the value to store at the supplied address.
  #define ACPP_PTX_STORE(NAME, SEM, TYPE, BITS, REG)          \
  inline void NAME(TYPE* ptr, TYPE value, ScopeSSCP scope)    \
  {                                                           \
    if (scope == ScopeSSCP::system)                           \
    {                                                         \
      asm volatile(                                           \
        "st." SEM ".sys.u" BITS " [%0], %1;" ::               \
        "l" (ptr), REG(value) :                               \
        "memory"                                              \
      );                                                      \
    }                                                         \
    else if (scope == ScopeSSCP::device)                      \
    {                                                         \
      asm volatile(                                           \
        "st." SEM ".gpu.u" BITS " [%0], %1;" ::               \
        "l" (ptr), REG(value) :                               \
        "memory"                                              \
      );                                                      \
    }                                                         \
    else                                                      \
    {                                                         \
      asm volatile(                                           \
        "st." SEM ".cta.u" BITS " [%0], %1;" ::               \
        "l" (ptr), REG(value) :                               \
        "memory"                                              \
      );                                                      \
    }                                                         \
  }
  ACPP_PTX_STORE(relaxed_store, "relaxed", __acpp_int32, "32", "r")
  ACPP_PTX_STORE(relaxed_store, "relaxed", __acpp_int64, "64", "l")
  ACPP_PTX_STORE(release_store, "release", __acpp_int32, "32", "r")
  ACPP_PTX_STORE(release_store, "release", __acpp_int64, "64", "l")
  #undef ACPP_PTX_STORE

  // Generates scoped 32-bit and 64-bit PTX compare-and-swap helpers.
  // Each operation atomically compares memory with the expected bit pattern,
  // conditionally installs the desired value, and returns the observed old value.
  // SEM selects relaxed, acquire, release, or acquire-release ordering.
  // Success/failure-order handling is performed by compare_exchange().
  #define ACPP_PTX_CAS(NAME, SEM, TYPE, BITS, REG)                              \
  inline TYPE NAME(TYPE* ptr, TYPE compare, TYPE desired, ScopeSSCP scope)      \
  {                                                                             \
    TYPE result;                                                                \
    if (scope == ScopeSSCP::system)                                             \
    {                                                                           \
      asm volatile(                                                             \
        "atom." SEM ".sys.cas.b" BITS " %0, [%1], %2, %3;" :                    \
        "=" REG(result) :                                                       \
        "l"(ptr), REG(compare), REG(desired) :                                  \
        "memory"                                                                \
      );                                                                        \
    }                                                                           \
    else if (scope == ScopeSSCP::device)                                        \
    {                                                                           \
      asm volatile(                                                             \
        "atom." SEM ".gpu.cas.b" BITS " %0, [%1], %2, %3;" :                    \
        "=" REG(result) :                                                       \
        "l"(ptr), REG(compare), REG(desired) :                                  \
        "memory"                                                                \
      );                                                                        \
    }                                                                           \
    else                                                                        \
    {                                                                           \
      asm volatile(                                                             \
        "atom." SEM ".cta.cas.b" BITS " %0, [%1], %2, %3;" :                    \
        "=" REG(result) :                                                       \
        "l"(ptr), REG(compare), REG(desired) :                                  \
        "memory"                                                                \
      );                                                                        \
    }                                                                           \
    return result;                                                              \
  }

  ACPP_PTX_CAS(relaxed_cas, "relaxed", __acpp_int32, "32", "r")
  ACPP_PTX_CAS(relaxed_cas, "relaxed", __acpp_int64, "64", "l")
  ACPP_PTX_CAS(acquire_cas, "acquire", __acpp_int32, "32", "r")
  ACPP_PTX_CAS(acquire_cas, "acquire", __acpp_int64, "64", "l")
  ACPP_PTX_CAS(release_cas, "release", __acpp_int32, "32", "r")
  ACPP_PTX_CAS(release_cas, "release", __acpp_int64, "64", "l")
  ACPP_PTX_CAS(acq_rel_cas, "acq_rel", __acpp_int32, "32", "r")
  ACPP_PTX_CAS(acq_rel_cas, "acq_rel", __acpp_int64, "64", "l")
  #undef ACPP_PTX_CAS

  // Selects the PTX atomic-load sequence for the requested SYCL order.
  // Relaxed and acquire use their corresponding scoped load instructions.
  // Sequential consistency uses an SC fence followed by an acquire load.
  // Unexpected orders trigger a device assertion in debug builds and use
  // the SC sequence as a conservative implementation fallback.
  template<class T>
  inline T load(T* ptr, OrderSSCP order, ScopeSSCP scope)
  {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,"The PTX primitive test targets 32/64-bit atomic types");
    switch (order) 
    {
      case OrderSSCP::relaxed:
        return relaxed_load(ptr, scope);
      case OrderSSCP::acquire:
        return acquire_load(ptr, scope);
      case OrderSSCP::seq_cst:
      default:
        debug_assert(order == OrderSSCP::seq_cst, "Invalid atomic load order", __FILE__, __LINE__, __func__);
        fence_seq_cst(scope);
        return acquire_load(ptr, scope);
    }
  }

  // Selects the PTX atomic-store sequence for the requested SYCL order.
  // Relaxed and release use their corresponding scoped store instructions.
  // Sequential consistency uses an SC fence followed by a release store,
  // which is stronger than the minimum PTX ABI store mapping.
  // Unexpected orders assert in debug builds and use the SC sequence.
  template<class T>
  inline void store(T* ptr, T value, OrderSSCP order, ScopeSSCP scope)
  {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,"The PTX primitive test targets 32/64-bit atomic types");
    switch (order) 
    {
      case OrderSSCP::relaxed:
        relaxed_store(ptr, value, scope);
        return;
      case OrderSSCP::release:
        release_store(ptr, value, scope);
        return;
      case OrderSSCP::seq_cst:
      default:
        debug_assert(order == OrderSSCP::seq_cst, "Invalid store order", __FILE__, __LINE__, __func__);
        fence_seq_cst(scope);
        release_store(ptr, value, scope);
        return;
    }
  }

  // Validates SYCL compare-exchange success and failure memory orders.
  // Converts their combined acquire/release requirements into one PTX
  // instruction order. Invalid pairs assert in debug builds and fall back
  // to sequential consistency when assertions are disabled.
  inline static constexpr OrderSSCP cas_singular_converter(OrderSSCP success, OrderSSCP failure)
  {
    bool valid = false;
    switch (success)
    {
        case OrderSSCP::relaxed:
        case OrderSSCP::release:
            valid = failure == OrderSSCP::relaxed;
            break;

        case OrderSSCP::acquire:
        case OrderSSCP::acq_rel:
            valid = failure == OrderSSCP::relaxed || failure == OrderSSCP::acquire;
            break;

        case OrderSSCP::seq_cst:
            valid = failure == OrderSSCP::relaxed || failure == OrderSSCP::acquire || failure == OrderSSCP::seq_cst;
            break;

        default:
            break;
    }
    if (!valid)
    {
      debug_assert(false, "Invalid PTX CAS success/failure orders", __FILE__, __LINE__, __func__);
      return OrderSSCP::seq_cst;
    }
    if (success == OrderSSCP::seq_cst)
    {
        return OrderSSCP::seq_cst;
    }
    const bool acquire = success == OrderSSCP::acquire || success == OrderSSCP::acq_rel || failure == OrderSSCP::acquire;
    const bool release = success == OrderSSCP::release || success == OrderSSCP::acq_rel;
    if (acquire && release)
        return OrderSSCP::acq_rel;

    if (acquire)
        return OrderSSCP::acquire;

    if (release)
        return OrderSSCP::release;

    return OrderSSCP::relaxed;
  }

  // Implements strong compare-exchange for aligned 32/64-bit atomic objects.
  // Selects one ordered PTX CAS instruction from the success/failure pair,
  // adding an SC fence for sequentially consistent operations.
  // Always updates *expected with the observed old value and returns true
  // exactly when the observed bits match the original expected value.
  template<class T>
  inline bool compare_exchange(
    T* ptr, T* expected, T desired, 
    OrderSSCP success, OrderSSCP failure, ScopeSSCP scope
  )
  {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,"The PTX primitive test targets 32/64-bit atomic types");

    const T compare = *expected;
    T observed;

    const OrderSSCP instruction_order = cas_singular_converter(success, failure);
    switch (instruction_order) 
    {
      case OrderSSCP::relaxed:
        observed = relaxed_cas(ptr, compare, desired, scope);
        break;
      case OrderSSCP::acquire:
        observed = acquire_cas(ptr, compare, desired, scope);
        break;
      case OrderSSCP::release:
        observed = release_cas(ptr, compare, desired, scope);
        break;
      case OrderSSCP::acq_rel:
        observed = acq_rel_cas(ptr, compare, desired, scope);
        break;
      case OrderSSCP::seq_cst:
      default:
        debug_assert(instruction_order == OrderSSCP::seq_cst, "Invalid cas order", __FILE__, __LINE__, __func__);
        fence_seq_cst(scope);
        observed =  acq_rel_cas(ptr, compare,  desired, scope);
        break;
    }

    *expected = observed;
    return observed == compare;
  }
}

#endif