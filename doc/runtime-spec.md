# AdaptiveCpp runtime specification

The AdaptiveCpp runtime library follows the requirements of a SYCL runtime library as described in the SYCL specification. The following specification assumes the SYCL specification, but expands on it where AdaptiveCpp provides stronger or slightly different guarantees.
It is assumed that the reader is at least familiar with the SYCL programming model.

## AdaptiveCpp buffer-accessor model

### Buffer behavior

#### Overview

A `buffer` is an object that provides storage of a fixed size, and makes that storage accessible on an arbitrary amount of devices. To this end, it manages allocations of the fixed buffer size on all devices where the buffer is accessed.

#### Persistent allocations

A goal of the `buffer` implementation is delivering predictable performance; as such all allocations managed by a `buffer` shall be of the fixed buffer size. No reallocations shall occur during the lifetime of the `buffer` without explicit user request, and managed allocations shall not be freed without explicit user request before `buffer` destruction. Once a `buffer` object has started to manage an allocation on a particular device, this allocation shall be used for all operations that access the `buffer` object on that device.
A pointer to buffer data obtained in a kernel shall be valid and point to the same memory for all subsequent kernels that are executed on the same device as long as the buffer object exists.

#### Explicit USM as foundation

Memory management operations of the `buffer` and storage shall be performed using SYCL 2020 USM pointers. This implies inherent interoperability between USM pointers and buffers. For example, if a pointer to a memory allocation managed by a `buffer` is obtained by the user, it shall behave like a USM pointer and USM operations shall work with that pointer as if it were a pointer obtained from a USM memory allocation function.

If `buffer` allocates memory, this shall be done using explicit USM allocations by default. If the `buffer` provides additional interoperability mechanisms that allow constructing buffers on top of user-provided USM pointers, those may be of other USM allocation types. In this case, the allocation shall still be interpreted by the `buffer` as a USM allocation that is bound to a single device.
For allocations on CPU backends, a `buffer` implementation may use USM host allocations (i.e. page-locked memory).

#### Mapping between allocations and devices

Allocations managed by a `buffer` shall not be shared between different physical devices; instead a buffer shall allocate individual memory buffers for each physical device on which it operates. This allows the scheduler and user to make stronger assumptions regarding necessary data migration and the performance impact of executing kernels simultaneously on different devices that read data from the same `buffer`. Allocations may only be shared between different SYCL devices that refer to the same physical hardware. For example, it may be desirable to have a single host allocation that is used by all CPU devices if there are multiple CPU backends available.

#### Allocation behavior

Memory shall be allocated lazily on a particular device when a `buffer` is first used on that device. However, some `buffer` constructors may require that data from a user-provided input pointer is copied to internal buffer storage. In this case `buffer` will perform an allocation in the constructor, typically on the host device, to hold that data.

#### Comments

* If a device pointer has been extracted from a buffer, it is valid at least until buffer destruction, and can be used for USM operations - provided the user manually synchronizes these USM operations with any operations the `buffer` is involved in.
* Because there are no partial allocations, accessing memory outside the bounds of a ranged accessor, but within the buffer bounds, is not undefined behavior. However, it is not guaranteed that this data is up-to-date and there might be other kernels operating on it simultaneously if the user does not manually synchronize (details below).


### Data transfers, accessors and dependencies

#### Data state tracking and pages

A SYCL implementation needs to track whether data stored in the `buffer` in an allocation on a particular device is up-to-date or outdated. This information allows it to determine whether the implicit requirements formulated by accessors need to be translated into actual data transfers.

In the AdaptiveCpp model, the range of the buffer is interpreted as a 3D grid that is divided into 3D chunks of fixed size in each dimension. These chunks will in the following be referred to as *pages* (unrelated to virtual memory pages of the operating system). An implementation may expose mechanisms that allow the user to set the page size in each dimension in the `buffer` constructor. The page size determines the granularity of memory management and data state tracking.

For each allocation managed on each device, the `buffer` implementation shall track for each page whether the data contained within the page is up-to-date or outdated.

If a page is fully contained within or overlaps with the accessed range of an accessor (taking into account the accessor's access offset and range), we use the terminology that the page is part of the accessor's *page range*.

Using an accessor that is not of a read-only access mode on a device *d*  shall cause all pages within its page range to be marked as outdated on all allocations except for those on device *d*. This is because the implementation has to assume that data was modified on `d`.

Data transfers generated from accessors (see below) shall cause transferred pages to be marked as up-to-date on the target allocation.

If a `buffer` is reinterpreted to a data type of different size than the original buffer element size or reshaped into a different range, the implementation may assume a page range for accessors to the reinterpreted buffer that is larger than the page range as defined above.

#### Data transfers

Accessors of `discard` access mode (`no_init` in SYCL 2020) shall never lead to data transfers.

Accessors referring to a `buffer` that does not contain any initialized data (e.g. because it was never written to and was not constructed with a user-provided input pointer) shall never lead to data transfers.

When a non-`discard` accessor is used on a particular device, a data transfer shall occur only if at least one of the pages within the accessor's page range is marked as outdated.

The implementation shall attempt to minimize both the number of transferred pages and the total number of backend data transfers, although the precise mechanism used and the detailed optimization criteria are implementation-defined.

#### Dependencies

Two accessors referring to the same `buffer` are considered *conflicting*, if one or both are not of read-only access mode and their page ranges overlap.

Two accessors referring to different `buffer` objects are never conflicting.

If two accessors are conflicting, a dependency is established between the command groups that they are used in. Dependent command groups are executed in submission order.

Independent command groups may be executed in parallel. For example, this includes the possibility of executing kernels in parallel on the same device, if this is supported by the backend and hardware.

#### Comments

* A smaller page size means a finer data management granularity; it may allow for more operations to be executed without dependencies in between them, but may also lead to a larger runtime overhead when tracking data state. The optimal page size is therefore a tradeoff. 
* Note that in the AdaptiveCpp model, subbuffers are neither needed, nor necessary, nor recommended to obtain parallel execution of kernels.
