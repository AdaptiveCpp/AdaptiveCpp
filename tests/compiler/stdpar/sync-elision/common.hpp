#pragma once

#define STDPAR_ENTRYPOINT \
    __attribute__((noinline)) __attribute__((annotate("hipsycl_stdpar_entrypoint")))

static int num_outstanding_operations = 0;

__attribute__((noinline))
extern "C" void __hipsycl_stdpar_optional_barrier() noexcept {
  num_outstanding_operations = 0;
}

STDPAR_ENTRYPOINT static void stdpar_call() {
  ++num_outstanding_operations;
  __hipsycl_stdpar_optional_barrier();
};

// This is a hack: If we just directly access num_outstanding_operations for testing,
// we trigger a load for this global variable, which would cause a barrier to be inserted.
// As a workaround, we introduce another STDPAR_ENTRYPOINT function to retrieve this value.
// This works, because the compiler skips calls to stdpar entrypoint functions when determining
// the next necessary synchronization.
STDPAR_ENTRYPOINT static int get_num_enqueued_ops() {
  return num_outstanding_operations;
};
