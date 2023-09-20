# Extension: Enqueue custom operation

This extension allows for efficient interoperability with the backend by exposing a mechanism that allows enqueuing custom asynchronous backend operations. AdaptiveCpp will treat these operations like kernels, and will ensure that they synchronize with other SYCL operations like a kernel would.
The main advantage over interoperability via SYCL 2020 host tasks is that a host task requires that the flow of execution returns to the host from device, and then potentially back to device again. This can add additional latency, and is not necessary if you only wish to enqueue additional backend operations. Custom operations allow enqueuing additional backend operations without having the execution flow return to the host.

Custom operations are submitted by providing a lambda or function object to `handler::hipSYCL_enqueue_custom_operation()`. This function object will be evaluated at DAG submission. It will be provided with an `interop_handle` as argument which is able to expose backend specific information, such as the backend queue (e.g. CUDA/HIP stream) that was selected for this operation.

Because custom operation function objects are evaluated at submission time (not DAG execution time), no host operations should be performed on input data inside the custom operation, since dependencies are not guaranteed to have completed yet. Similarly, no synchronous operations should be submitted to the backend.

Only asynchronous operations operating on the target device from the backend are guaranteed to behave correctly. It is also necessary to submit custom operations only to the backend queue (e.g. CUDA stream) provided by the `interop_handle`. This is because AdaptiveCpp will assume that any subsequent SYCL operations can synchronize with the custom operation by synchronizing with the backend queue.

## host task vs enqueuing custom operations for interoperability

Use a host task when
* You want to execute work on the host
* You want input data in a well defined state when your function object is evaluated
* You want to run synchronous backend operations that depend on when they are executed during DAG execution
* You are not sure if your use case is safe for the the custom operation mechanism and want to be safe
* You don't mind introducing synchronization between host and device

Use a custom operation when
* You want to submit additional asynchronous tasks to the backend
* You care about latency.

## API reference

```c++
// Enqueue a custom operation. f is a callable of signature void(interop_handle).
template <class InteropFunction>
void handler::hipSYCL_enqueue_custom_operation(InteropFunction f);

// Equivalent queue shortcuts are available as well for USM use cases
template<class InteropFunction>
event queue::hipSYCL_enqueue_custom_operation(InteropFunction op);

template <class InteropFunction>
event queue::hipSYCL_enqueue_custom_operation(InteropFunction op, event dependency):

template <class InteropFunction>
event queue::hipSYCL_enqueue_custom_operation(InteropFunction op,
                                              const std::vector<event> &dependencies);

```

## Example

This example uses the HIP backend, other backends such as CUDA work similarly.
```c++
sycl::queue q;

q.submit([&](sycl::handler &cgh) {
    auto acc = some_buff.get_access<sycl::access::mode::read>(cgh);

    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
      // Can extract device pointers from accessors
      void *native_mem = h.get_native_mem<sycl::backend::hip>(acc);
      // Can extract stream (note: get_native_queue() may not be 
      // supported on CPU backends)
      hipStream_t stream = h.get_native_queue<sycl::backend::hip>();
      // Can extract HIP device (note: get_native_device() may not be
      // supported on CPU backends)
      int dev = h.get_native_device<sycl::backend::hip>();
      // Can enqueue arbitrary backend operations. This could also be a kernel launch
      // or a call to a library that enqueues operations on the stream etc
      hipMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
                      hipMemcpyDeviceToHost, stream);
    });
  });

q.wait();

```