# Environment variables used by hipSYCL

* `HIPSYCL_DEBUG_LEVEL` - if set, overrides the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is the value of `HIPSYCL_DEBUG_LEVEL` [macro](macros.md).
* `HIPSYCL_VISIBILITY_MASK` - can be used to activate only a subset of backends. Syntax: `backend;backend2;..`. Possible values are `omp`, `cuda`, `hip` and `ze` (as level zero is the backend and `spirv` in `HIPSYCL_TARGETS` just the target format). `omp` will always be active as a CPU backend is required. Device level visibility has to be set via vendor specific variables for now, including `{CUDA,HIP}_VISIBLE_DEVICES` and `ZE_AFFINITY_MASK`.
