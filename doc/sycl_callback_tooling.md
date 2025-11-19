# Sycl callback tooling 

To enable tracing, benchmarking, and simple debugging of sycl applications, this fork of AdaptiveCpp experimentally supports a callback mechanism.  


## Usage: 

### Registering Callbacks (User side)
The user can start tracers by setting the environment variable `SYCL_TOOL_LIBRARY` to the paths to tooling libraries separated by a colon `:`. 

### Implementation of Callbacks (Tooling side)
A tracer can be implemented as a dynamically loaded library. (shared object `.so` on Linux, dynamic link library `.dll` on Windows and `.dylib` on macos). The implementer then has the option to define one or more of the supported callbacks. Furthermore, a function with the signature `void init_register()` must be defined. This function is called when the library is loaded and can be used to initialize the tracer. The loading utility assumes unmangled C function names, so the implementer must ensure that the `init_register`-function name is not mangled (e.g. by using `extern "C"` in a C++ declaration). Within the `init_register`-function the implementer can register the desired callbacks by calling the function
`init_##CallbackName(void (*callback)(...))`, where `CallbackName` is the name of the callback to register and `callback` is a pointer to the function implementing the callback. For example, to register a callback for the `submit_start` event, the implementer would call `init_submit_start(&my_submit_start_callback);` within the `init_register`-function, where `my_submit_start_callback` is a function with the signature `void my_submit_start_callback(void* state)`. Furthermore, the function call `void init_state(void* state)` can be used to register a pointer to a user defined state. This pointer is then passed as the first argument to all the callbacks. This can be used to maintain state information across multiple callback invocations. The implementer can then typecast the `state` pointer to the appropriate type within the callback implementations.

**Note**: It is recommended that the tracer state is allocated on the heap, i.e. using `new` or `malloc` as stack-allocated states may cause undefined behavior during shutdown of the program. The state can be created within the `init_register`-function. 

Example of a simple tracer implementation:

```cpp 
#include <iostream>
#include "hipSYCL/sycl/tracer_utils.hpp"
    
struct MyTracerState {
    ...
    // Add any state information needed for the tracer
    ...
};
    
extern "C" {
    void my_submit_start_callback(void* state) {
        (MyTracerState*)tracer_state = (MyTracerState*)state;
        ...
        // Do some stuff with tracer state pointer
        ...
        std::cout << "Submit started!" << std::endl;
    }

    void init_register() {
        MyTracerState* tracer_state = new MyTracerState();
        init_state(tracer_state);
        init_submit_start(my_submit_start_callback);
    }
}
```

For a complete list of supported callbacks, signatures, and their description, see the table in section [Callback Signatures and Trace Records](#callback-signatures-and-trace-records).


## Callback Signatures and Trace Records

The following callbacks are supported:

| Callback Name                 | Signature                                              | Description (Called at...)            |
|-------------------------------|--------------------------------------------------------|---------------------------------------|
| queue_impl_constructor        | `void(void* state, size_t queue_hash, bool is_inorder)`| Construction of SYCL queue  |
| queue_impl_destructor         | `void(void* state, size_t queue_id)`                   | Destruction of SYCL queue   |
| dag_node_constructor          | `void(void* state, size_t node_id)`                    | Construction of a SYCL event|
| dag_node_destructor           | `void(void* state, size_t node_id)`                    | Destruction of a SYCL event |
| submit_start                  | `void(void* state)`                                    | the start of a queue::submit <br>operation|
| submit_end                    | `void(void* state, size_t event_hash, size_t queue_id)`| the end of a queue::submit operation <br>(but not at submission with secondaryQueue, see below)|
| submit_secondary_start        | `void(void* state)`                                    | start of a secondary submit operation|
| submit_secondary_end          | `void(void* state, size_t event_hash, size_t queue_id)`| end of a secondary submit operation|
| parallel_for_start            | `void(void* state)`                                    | start of a parallel_for operation|
| parallel_for_end              | `void(void* state)`                                    | end of a parallel_for operation|
| parallel_for_work_group_start | `void(void* state)`                                    | start of a parallel_for_work_group operation|
| parallel_for_work_group_end   | `void(void* state)`                                    | end of a parallel_for_work_group operation|
| single_task_start             | `void(void* state)`                                    | start of a single_task operation|
| single_task_end               | `void(void* state)`                                    | end of a single_task operation|
| memcpy_start                  | `void(void* state)`                                    | start of a memcpy operation|
| memcpy_end                    | `void(void* state)`                                    | end of a memcpy operation|
| wait_start                    | `void(void* state)`                                    | start of a wait operation|
| wait_queue_end                | `void(void* state, size_t queue_id)`                   | end of a queue::wait operation|
| wait_event_end                | `void(void* state, size_t event_id)`                   | end of an event::wait operation|
| memset_start                  | `void(void* state)`                                    | start of a memset operation|
| memset_end                    | `void(void* state)`                                    | end of a memset operation|
| fill_start                    | `void(void* state)`                                    | start of a fill operation|
| fill_end                      | `void(void* state)`                                    | end of a fill operation|
| copy_start                    | `void(void* state)`                                    | start of a copy operation|
| copy_end                      | `void(void* state)`                                    | end of a copy operation|
| malloc_device_start           | `void(void* state)`                                    | start of a malloc_device operation|
| malloc_device_end             | `void(void* state, void* ptr)`                         | end of a malloc_device operation|
| malloc_host_start             | `void(void* state)`                                    | start of a malloc_host operation|
| malloc_host_end               | `void(void* state, void* ptr)`                         | end of a malloc_host operation|
| malloc_shared_start           | `void(void* state)`                                    | start of a malloc_shared operation|
| malloc_shared_end             | `void(void* state, void* ptr)`                         | end of a malloc_shared operation|
| free_start                    | `void(void* state)`                                    | start of a free operation|
| free_end                      | `void(void* state, void* ptr)`                         | end of a free operation|
| finalize                      | `void(void* state)`                                    | shutdown of the SYCL runtime|

If multiple tooling libraries are loaded, the order of initialization is the same as the order of the path list in the `SYCL_TOOL_LIBRARY` environment variable. The order of finalization is the inverse order of initialization. 

**Note**: In AdaptiveCpp, the SYCL runtime is a singleton associated with the existence of a sycl::queue, i.e. the runtime exists as long as there is at least one sycl::queue. 
If no sycl::queue is created, the runtime is not initialized and thus the tooling library is not loaded. Furthermore, if there is a point in the program at which there is no sycl::queue, the runtime is finalized and the tracers are finalized as well. When a queue is created again, the runtime is reinitialized and so is the tracer.

