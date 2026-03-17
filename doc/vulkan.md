# Vulkan

Experimental Vulkan compute backend. Backend devices require a Vulkan 1.3 or later
physical device although it was testing using the LunarG SDK 1.4 versions.

The following physical device features are also required to be
support for a Vulkan device to be available although future work would be
to make checking for shader capabilities more lazy and only
error if you try to use a kernel that doesn't have the capability
enabled.

* `bufferDeviceAddress`
* `timelineSemaphore`
* `shaderInt8`
* `shaderFloat16`
* `storagePushConstant8`
* `variablePointers`
* `variablePointersStorageBuffer`
* `storagePushConstant16`
* `shaderInt16`
* `shaderInt64`

The Vulkan validation layer is enabled in debug builds.

Vulkan consumes SPIR-V kernels which are required to be match the Vulkan SPIR-V
environment requirements, rather than the OpenCL SPIR-V environment requirements like
for the Level-Zero and OpenCL backends. In order to turn SYCL kernels into Vulkan
consumable SPIR-V the [clspv](github.com/google/clspv) tool is used, which is a
requirement for building AdaptiveCpp with the Vulkan backend.

## Building

The LunarG Vulkan SDK is a dependency for building the Vulkan backend, as it
provides the Vulkan layers & loader along with SPIR-V Tools. A `clspv` binary
that is detectable by `find_program` is also a dependency.

Example CMake invocation
```sh
cmake -DWITH_VULKAN_BACKEND=ON -DCMAKE_PROGRAM_PATH=<path/to/clspv/build/dir>
```

## Design

### Runtime

#### USM

The use of the [VK_KHR_buffer_device_address](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_buffer_device_address.html)
extension is a key part of the implementation as it allows us to implement
device USM. By using `vkGetBufferDeviceAddress` we can get a device addressable to
give to the user that they can use in a kernel. However, host/shared USM isn't
supportable as this pointer isn't valid on host, the underlying Vulkan buffer needs
memory mapped to a make that possible which is an intermediate step that the
SYCL runtime performs when the user invokes a SYCL API using a device USM pointer.

In order to implement `memcpy` using the asynchronous command-buffer API, we
need to use the VK buffer objects tied to those device addressable USM
pointers. However, if the source or destination operand is a host pointer
then there won't be an associated VK buffer. We therefore need to create
temporary VK buffers for these operands which are tied to the lifetime
of the async command. If a source operand is a host pointer, then
we do a memcopy from the host pointer into the temporary VK buffer
before submitting the async operation. When the destination operand
is a host pointer we submit an async host thread that executes
after the command-buffer has completed which copies the temporary
VK buffer back to the host pointer.

In future `VK_KHR_device_address_commands` could be used to implement
`memcpy`.

#### Compute Pipeline

A compute pipeline is created for each kernel JIT compiled for a specific workgroup size.
As we are JIT compiling kernels, each translation unit that the clspv compiler sees only
contains a single kernel in it, and therefore only use a single descriptor set and binding
when creating the kernels.

When a kernel is re-submitted in the SYCL application the existing `vk_executable_object`
is found in the cache with the JIT compiled SPIR-V kernel code. This will already
have a `vk_kernel_object` attached representing the kernel function in that binary.
If the workgroup size of the kernel enqueue is the same as previously then the
old cached `vk_kernel_pipeline` instance will be used with the Vulkan compute pipeline created
for that specific workgroup size. Otherwise, if the workgroup size is different, a
new `vk_kernel_pipeline` object will be created to instantiate a Vulkan compute pipeline
for that workgroup size. However, the AdaptiveCpp SSCP compiler seem to create a
new executable for each work-group size with the information embedded in the required
work-group size metadata so in practice a new `vk_executable_object` is created for
each work-group size variant of a kernel.

For small numbers of kernel arguments, no buffer descriptors may be required and all
arguments can be set pushing push constants before the invocation of the Vulkan
command-buffer containing the kernel. For larger numbers of kernel arguments or struct
arguments that are decomposed, a single uniform buffer is used with binding 0,
and each argument is an offset into that uniform buffer.

#### Queue

Vulkan devices are conceptually split into physical devices and logical devices, where
a logical devices is created by the user based on the available physical devices
and their capabilities. During the creation of the logical device the SYCL runtime
finds the first Vulkan queue family which supports compute, and creates a single
single queue to tie to the logical device. All `sycl::queue` objects created map
to that single queue, but each have control their own command-buffers to use
with that queue.

Each queue submission is in it's own command-buffer, recorded with a single-submit bit.
And incrementing a timeline semaphore owned by the queue such that the submission depends
on the previous submission and the signal value can be used to identify the command.
This combination of a signal value and queue uniquely identifying a command leads them
to the implementation basis of the event class.

#### Host Threads

Not all SYCL commands are implementation in command-buffer commands, e.g. `memset`
needs to work with less than 4-bytes. In which case asynchronous host threads are
used to perform the work, waiting for and signalling the queue semaphore from host.

Host threads are also used to perform work that needs to be done after the device
command has completed, but before we tell other SYCL commands the DAG node has
completed. For example, freeing temporary allocations created to implement
memcopy.

### Compiler

Uses SPIR-V version 1.3 for the `GroupNonUniform` capability.

#### clspv

Kernels are compiled by using the LLVM-IR input to clspv to create Vulkan consumable SPIR-V.
This involves the SSCP compiler flavouring the LLVM-IR to resemble OpenCL-C generated IR,
which is then lowered appropriately by clspv. Transformations are defined in the
`llvm-to-clspv` tool to achieve this.

The SPIR-V reflection non-semantic instructions are then used to provide the
Vulkan runtime with the appropriate information on how to set the kernel arguments. For
example as push constants, or via a uniform buffer.

See clspv doc on [OpenCL-C restrictions](https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md#opencl-c-restrictions)
for inherited restrictions, such as no double precision floating point support.

#### Physical Addressing

A key part of the SPIR-V generated is that we
can request `clspv` uses the `Physical64` memory model, as opposed to Logical addressing,
which allows for the use of device USM allocated pointers.

#### Specialization constants

A specialization constant will be created by `clspv` if the required workgroup size attribute
was not set in the kernel IR, which is part of the reflection information that the runtime
can detect and set.

## TODO

* Complete lib kernel implementation
* Test with real application
* Try to make compiler passes less brittle
* Improve device-info queries marked with TODO comments
* Lazily error on use of kernels which aren't supported by physical device, rather
  than not reporting VK physical device at all as a SYCL device.

## Test Status

No CI is yet setup, tested sycl test suite locally on AMD pheonix integrated GPU and AMD MI100 & MI200 discrete GPUs.
MI200 has several fails that need investigated (See [issue 9](#issue-9)) but otherwise the fails can be categorize
as:

| Suite                     | Status |
| ------------------------- | ------ |
| `accessor_tests`          | [Issue 1](#issue-1), [Issue 4](#issue-4) & [Issue 8](#issue-8)  |
| `atomic_tests`            | `fetch_ops` [Issue 2](#issue-2)   |
| `buffer_tests`            | Pass   |
| `explicit_copy_tests`     | Pass   |
| `extension_tests`         | Pass   |
| `fill_tests`              | Pass   |
| `group_functions_tests`   | [Issue 5](#issue-5) & [Issue 10](#issue-10) |
| `half_tests`              | Pass   |
| `id_range_tests`          | Pass   |
| `info_queries`            | Pass   |
| `interop_handle_tests`    | Pass   |
| `item_tests`              | Pass   |
| `kernel_invocation_tests` | Pass   |
| `math_tests`              | Pass   |
| `marray_tests`            | `short3` [Issue 3](#issue-3)    |
| `profiler_tests`          | Pass    |
| `reduction_tests`         | [Issue 6](#issue-6)   |
| `reference_semantics`     | Pass   |
| `rel_tests`               | Pass   |
| `sub_group_tests`         | Pass   |
| `usm_tests`               | Pass   |
| `vec_tests`               | Pass   |
| `queue_tests`             | Pass   |
| `multi_ptr_test_suite`    | Pass   |
| `smoke_task_queue`        | [Issue 7](#issue-7) & [Issue 8](#issue-8) |

### Issue 1

Only memset instructions using zero initializers are lowered by clspv.
This affects `ranged_accessor_tests/ranged_accessor_1d_iterator` which creates a memset with `-1`.

### Issue 2

clspv [doesn't support floating point atomics](https://github.com/google/clspv/issues/392#issuecomment-503236450).
This currently generates incorrect code rather than throwing the user a nice error when
floating point atomics are used in kernel code.

### Issue 3

clspv can't deal with `i48` LLVM IR types generated from `marray<short, 3>` testing

### Issue 4

`accessor_api` issue with clspv consuming the LLVM-IR used to implement the atomic exchange builtin. It looks
like this is due to the builtin taking a volatile pointer to a address space attribute pointer.
Idea to fix is to add `__attribute__(addressspace(x))` to libkernel builtin definition, then
do a clspv compiler pass that gets rid of the volatile load and do a manual mem2reg transform.

### Issue 5

Workgroup functions not yet implemented in libkernel for clspv.

### Issue 6

Correctness issues that fail verification, needs further investigation.

### Issue 7

`queue_offset_subtract` fails with either incorrect results or where
device is lost during kernel execution.

Same issue can also be same using `clvk` with the following OpenCL kernel,
and using physical addressing with `spir64` in the clspv invocation.

```c
kernel void test_simple(global int* out, global int* in, int offset)
{
    size_t gid = get_global_id(0);
    out[gid] = in[offset-1];
}
```

Investigations so far:
 * Removing `-1` fixes the issue
 * Using `volatile* to keep the load but remove the store fixes the issue`
 * Doing `out[id] = &in[offset-1] - &in` gives the correct offset
 * Generating logical memory model spirv from than physical addressing SPIRV gives the correct
   result in clvk

### Issue 8

> AMD MI GPU Only

Non-deterministic verification fail in `accessors_tests/local_accessors` and
`smoke_task_queue_tests/queue_local` on MI100 and MI200, needs further investigation.

### Issue 9

> AMD MI200 GPU only

Assorted MI200 test fails that needs further investigation:
* `accessor_tests/nested_subscript`
* `accessor_tests/offset_2d`
* `accessor_tests/offset_nested_subscript`
* `sub_group_tests/sub_group`
* `smoke_task_queue_tests/queue_local`
* `relational_tests/rel_genfloat_unary`
* `fill_tests/*`
* `item_api_tests/*`
* `extension_tests/buffer_page_size`
* `explicit_copy/explicit_buffer_copy_host_ptr`

### Issue 10

> Issue only appears in RADV Phoenix

Workgroup size of larger than 128 leads to errors in `group_functions_tests/group_barrier`
where the result indexes above thread id 128 is no longer synchronized with
the previous threads.

I could also reproduce this error with clvk on the same system using the
following kernel reduced from the group_barrier test.

```c
kernel void test_simple(global int* acc)
{
    int tmp          = -10000;
    int local_id     = get_local_id(0);
    int local_size   = get_local_size(0);
    for (int i = 0; i < local_size; ++i) {
        if (local_id == i) {
          for (int j = 0; j < 10000; ++j)
            tmp++;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (local_id == i)
          acc[i] = tmp;
        barrier(CLK_GLOBAL_MEM_FENCE);
        tmp = acc[i];
    }
}
```
