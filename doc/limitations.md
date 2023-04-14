# Limitations

## Unimplemented features 
The following is a (probably incomplete) list of features that are not yet implemented in hipSYCL
* hierarchical parallel for: flexible work group ranges are unsupported (hierarchical dispatch with ranges fixed at `parallel_for_work_group` invocation is supported).
* hierarchical parallel for: Within work group scope execution may not be limited to only one work item per group.
* hierarchical parallel for: Initializing variables in work group scope may not work correctly
* Some builtins are unimplemented, e.g. `sycl::native` functions
* Images
* vec<> class lacks convert(), as(), swizzled temporary vector objects lack operators
* Error handling: wait_and_throw() and throw_asynchronous() do not invoke async handler
* 0-dimensional objects (e.g 0D accessors) are mostly unimplemented
* SYCL 1.2.1: Because hipSYCL is not based on OpenCL, all SYCL OpenCL interoperability features are unimplemented.

#### Other limitations
* If the SYCL namespace is fully opened with a `using namespace cl::sycl` statement, name collisions *may* occur since the SYCL spec requires the existence of SYCL vector types such as `float4` in the SYCL namespace with the same name as CUDA/HIP vector types which live in the global namespace.