# Explicit buffer policies

`sycl::buffer` objects can have a variety of different behaviors, sometimes depending on subtle differences in how they were constructed.
For example, whether a writeback occurs depends on the combination of constness of the buffer's pointer argument, constness of buffer type and potentially buffer properties.
This can potentially make it error-prone to use. Errors can either lead to undesired behavior affecting correctness (e.g. no writeback when it *was* expected) or performance issues (e.g. writeback when it *was not* expected).

AdaptiveCpp therefore introduces explicit buffer policies that allow the programmer to explicitly specify the desired buffer behavior. Additionally, AdaptiveCpp adds certain buffer behaviors not covered by the SYCL specification.

Buffer policies can be enabled by passing buffer policy property to the `property_list` of the `buffer` constructor.

## Available buffer policies

| Policy property | Behavior if set to true | Behavior if set to false |
| ------ | ----- | ----- |
| `property::buffer::AdaptiveCpp_buffer_uses_external_storage(bool)` | Instruct the buffer to act as view and operate directly on provided input pointers. Pointer should not be used by the user during the lifetime of the buffer. | Only use supplied pointer as input argument and copy data to internal storage. Pointer can be used by used as desired after buffer construction. |
| `property::buffer::AdaptiveCpp_buffer_writes_back(bool)` | Submit writeback to supplied user pointer at buffer destruction. If no data needs to be transferred, the writeback may be optimized away by the runtime. | Do not submit writeback in destructor. |
| `property::buffer::AdaptiveCpp_buffer_destructor_blocks(bool)` | buffer destructor blocks until all tasks operating on the buffer have completed. | buffer destructor does not block. The user is responsible for making sure that all kernels and other operations working on the buffer are synchronized, e.g. using `queue::wait()` or `event::wait()`. It is allowed for the buffer to be destroyed before operations have completed because operations will retain references to the buffer's data storage. |

To describe buffers and their behavior better, AdaptiveCpp adopts the following terminology:

| | Destructor blocks? | Writes back ? | Uses external storage? |
| ----- | ----- | ----- | ----- |
| yes | `sync_` | `_writeback_` | `view` |
| no  | `async_` | - | `buffer` |

For example, a `sync_view` blocks in the destructor, does not write back and operates directly on provided input pointers. An `async_writeback_view` does not block in the destructor, writes data back, and operates directly on provided input pointers.

These are logical data types, they are still expressed using regular `buffer<T>` objects, but they have dedicated factory functions to simplify construction.

## API reference

```c++
namespace sycl {
namespace property::buffer {

struct AdaptiveCpp_buffer_uses_external_storage {
  AdaptiveCpp_buffer_uses_external_storage(bool toggle);
};

struct AdaptiveCpp_buffer_writes_back {
  AdaptiveCpp_buffer_writes_back(bool toggle);
};

struct AdaptiveCpp_buffer_destructor_blocks {
  AdaptiveCpp_buffer_destructor_blocks(bool toggle);
};

} // namespace property buffer


/// Only uses internal storage, no writeback, blocking destructor
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_buffer(sycl::range<Dim> r);

/// Only uses internal storage, no writeback, blocking destructor
/// Data pointed to by ptr is copied to internal storage.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_buffer(const T* ptr, sycl::range<Dim> r);

/// Only internal storage, no writeback, non-blocking destructor
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_buffer(sycl::range<Dim> r);

/// Only internal storage, no writeback, non-blocking destructor
/// Data pointed to by ptr is copied to internal storage.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_buffer(const T* ptr, sycl::range<Dim> r);

/// Uses provided storage, writes back, blocking destructor
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_writeback_view(T* host_view_ptr, sycl::range<Dim> r);

/// Uses provided storage, writes back, non-blocking destructor
/// Directly operates on host_view_ptr.
/// The queue can be used by the user to wait for the writeback to complete.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_writeback_view(T* host_view_ptr, sycl::range<Dim> r, const sycl::queue& q);

/// Uses provided storage, writes back, non-blocking destructor
/// Directly operates on host_view_ptr.
/// Note: Because the writeback is asynchronous, there is no way
/// for the user to wait for its completion. In most cases, users
/// should use make_async_writeback_view(T*, sycl::range<Dim>, const sycl::queue&)
/// instead.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_writeback_view(T* host_view_ptr, sycl::range<Dim> r);

/// Uses provided storage, does not write back, blocking destructor
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_view(T* host_view_ptr, sycl::range<Dim> r);

/// Uses provided storage, does not write back, non-blocking destructor
/// Directly operates on host_view_ptr.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_view(T* host_view_ptr, sycl::range<Dim> r);

/// USM interoperability types

/// Uses provided storage, does not write back, blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_view(
    const std::vector<buffer_allocation::tracked_descriptor<T>>
        &input_allocations,
    sycl::range<Dim> r);

/// Uses provided storage, does not write back, non-blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_view(
    const std::vector<buffer_allocation::tracked_descriptor<T>>
        &input_allocations,
    sycl::range<Dim> r);

/// same as make_sync_view(
//     std::vector<buffer_allocation::tracked_descriptor<T>>&, sycl::range<Dim>)
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_usm_view(const std::vector<buffer_allocation::tracked_descriptor<T>>
                       &input_allocations,
                   sycl::range<Dim> r);

/// same as make_async_view(
//     std::vector<buffer_allocation::tracked_descriptor<T>>&, sycl::range<Dim>)
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_usm_view(const std::vector<buffer_allocation::tracked_descriptor<T>>
                       &input_allocations,
                    sycl::range<Dim> r);


}

```

## Example

```c++

int* data1 = ...;
int* data2 = ...;
std::size_t size = ...;

sycl::queue q;

// Because these are async writeback types, need to provide a queue
// so we can wait for the writeback
{
  auto v1 = sycl::make_async_writeback_view(data1, sycl::range{size}, q);
  auto v2 = sycl::make_async_writeback_view(data2, sycl::range{size}, q);

  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> a1 {v1, cgh};
    sycl::accessor<int> a2 {v2, cgh};

    cgh.parallel_for(size, [=](sycl::id<1> idx){
      a1[idx] += a2[idx];
    });
  });
} // Asynchronous writebacks triggered here

// At some point, wait for writebacks to complete
q.wait();

```
