# Environment variables used by Open SYCL

* `HIPSYCL_DEBUG_LEVEL`: if set, overrides the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is the value of `HIPSYCL_DEBUG_LEVEL` [macro](macros.md).
* `HIPSYCL_VISIBILITY_MASK`: can be used to activate only a subset of backends. Syntax: `backend;backend2;..`. Possible values are `omp`, `cuda`, `hip` and `ze` (as level zero is the backend and `spirv` in `HIPSYCL_TARGETS` just the target format). `omp` will always be active as a CPU backend is required. Device level visibility has to be set via vendor specific variables for now, including `{CUDA,HIP}_VISIBLE_DEVICES` and `ZE_AFFINITY_MASK`.
* `HIPSYCL_RT_DAG_REQ_OPTIMIZATION_DEPTH`: maximum depth when descending the DAG requirement tree to look for DAG optimization opportunities, such as eliding unnecessary dependencies.
* `HIPSYCL_RT_MQE_LANE_STATISTICS_MAX_SIZE`: For the `multi_queue_executor`, the maximum size of entries in the lane statistics, i.e. the maximum number of submissions to retain statistical information about. This information is used to estimate execution lane utilization.
* `HIPSYCL_RT_MQE_LANE_STATISTICS_DECAY_TIME_SEC`: The time in seconds (floating point value) after which to forget information about old submissions.
* `HIPSYCL_RT_SCHEDULER`: Set scheduler type. Allowed values: 
    * `direct` is a low-latency direct-submission scheduler. 
    * `unbound` is the default scheduler and supports automatic work distribution across multiple devices. If the `HIPSYCL_EXT_MULTI_DEVICE_QUEUE` extension is used, the scheduler must be `unbound`.
* `HIPSYCL_DEFAULT_SELECTOR_BEHAVIOR`: Set behavior of default selector. Allowed values:
    * `strict` (default): Strictly behave as defined by the SYCL specification
    * `multigpu`: Makes default selector behave like a multigpu selector from the `HIPSYCL_EXT_MULTI_DEVICE_QUEUE` extension
    * `system`: Makes default selector behave like a system selector from the `HIPSYCL_EXT_MULTI_DEVICE_QUEUE` extension
* `HIPSYCL_HCF_DUMP_DIRECTORY`: If set, hipSYCL will dump all embedded HCF data files in this directory. HCF is hipSYCL's container format that is used by all compilation flows that are fully controlled by hipSYCL to store kernel code.
* `HIPSYCL_PERSISTENT_RUNTIME`: If set to 1, hipSYCL will use a persistent runtime that will continue to live even if no SYCL objects are currently in use in the application. This can be helpful if the application consists of multiple distinct phases in which SYCL is used, and multiple launches of the runtime occur.
* `HIPSYCL_RT_MAX_CACHED_NODES`: Maximum number of nodes that the runtime buffers before flushing work.
* `HIPSYCL_SSCP_FAILED_IR_DUMP_DIRECTORY`: If non-empty, hipSYCL will dump the IR of code that fails SSCP JIT into this directory.