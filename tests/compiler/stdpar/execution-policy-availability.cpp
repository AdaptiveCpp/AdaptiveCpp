// RUN: %acpp %s --acpp-targets=generic -std=c++17 -E -o %t \
// RUN:   -I%S/../../../include -Xclang -verify
// RUN: %acpp %s --acpp-targets=generic -std=c++17 -E -o %t \
// RUN:   -I%S/../../../include -DTEST_LEGACY_PSTL_DISABLED -Xclang -verify
// RUN: %acpp %s --acpp-targets=generic -std=c++17 -E -o %t \
// RUN:   -I%S/../../../include -DTEST_PSTL_ENABLED=0 -Xclang -verify
// RUN: %acpp %s --acpp-targets=generic -std=c++17 -E -o %t \
// RUN:   -I%S/../../../include -DTEST_PSTL_ENABLED=1 -Xclang -verify

#include <execution>
#include <type_traits>

// Preprocess the availability check with both libc++ configuration conventions,
// regardless of the host standard library. Load system headers before overriding
// their macros so these test settings affect only the AdaptiveCpp header.
#ifndef _LIBCPP_VERSION
#define _LIBCPP_VERSION 1
#endif
#undef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
#undef _LIBCPP_HAS_EXPERIMENTAL_PSTL

#ifdef TEST_LEGACY_PSTL_DISABLED
#define _LIBCPP_HAS_NO_INCOMPLETE_PSTL
#elif defined(TEST_PSTL_ENABLED)
#define _LIBCPP_HAS_EXPERIMENTAL_PSTL TEST_PSTL_ENABLED
#endif

#if defined(TEST_LEGACY_PSTL_DISABLED) || \
    (defined(TEST_PSTL_ENABLED) && !TEST_PSTL_ENABLED)
// expected-error@* {{AdaptiveCpp stdpar requires libc++ execution policies; try -fexperimental-library}}
#else
// expected-no-diagnostics
#endif

#include <hipSYCL/std/stdpar/detail/execution_fwd.hpp>
