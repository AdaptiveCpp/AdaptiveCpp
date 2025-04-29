// Taken from https://reviews.llvm.org/D149364

// CUDA headers define __noinline__ which interferes with libstdc++'s use of
// `__attribute((__noinline__))`. In order to avoid compilation error,
// temporarily unset __noinline__ when we include affected libstdc++ header.
// this is an additional issue when using gcc13 headers

#pragma push_macro("__noinline__")
#undef __noinline__
#include_next "bits/basic_string.tcc"

#pragma pop_macro("__noinline__")
