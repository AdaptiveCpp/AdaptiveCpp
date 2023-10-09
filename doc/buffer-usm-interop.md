# AdaptiveCpp buffer-USM interoperability

AdaptiveCpp supports interoperability between `buffer` objects and USM pointers. All `buffer` memory allocations are USM pointers internally that can be extracted and be used with USM operations.
Similarly, `buffer` objects can be constructed on top of existing USM pointers.

AdaptiveCpp follows its own [specification](runtime-spec.md) for the AdaptiveCpp buffer-accessor model. Refer to this document to understand the memory management and allocation behavior of AdaptiveCpp buffer objects. Using buffer-USM interoperability without a solid understanding of AdaptiveCpp's buffer model is not recommended.

## Buffer introspection

USM pointers that a buffer currently manages can be queried, and some aspects can be modified.

## API reference

```c++
namespace sycl {

namespace buffer_allocation {
/// Describes a buffer memory allocation represented by a USM pointer 
/// and additional meta-nformation.
template <class T> struct descriptor {
  /// the USM pointer
  T *ptr;
  /// the device for which this allocation is used.
  /// Note that the runtime may only maintain
  /// a single allocation for all host devices.
  device dev;
  /// If true, the runtime will delete this allocation
  /// at buffer destruction.
  bool is_owned;
};

}

template <typename T, int dimensions,
          typename AllocatorT>
class buffer {
public:

  /// Iterate over all allocations used by this buffer, and invoke
  /// a handler object for each allocation.
  /// \param h Handler that will be invoked for each allocation.
  ///  Signature of \c h: void(const buffer_allocation::descriptor<T>&)
  template <class Handler>
  void for_each_allocation(Handler &&h) const;

  /// Get USM pointer for the buffer allocation of the specified device.
  /// \return The USM pointer associated with the device, or nullptr if
  /// the buffer does not contain an allocation for the device.
  T *get_pointer(const device &dev) const;

  /// \return Whether the buffer contains an allocation for the given device.
  bool has_allocation(const device &dev) const;

  /// \return the buffer allocation object associated with the provided
  /// device. If the buffer does not contain an allocation for the specified
  /// device, throws \c invalid_parameter_error.
  buffer_allocation::descriptor<T> get_allocation(const device &dev) const;

  /// \return the buffer allocation object associated with the provided pointer.
  /// If the buffer does not contain an allocation described by ptr,
  /// throws \c invalid_parameter_error.
  buffer_allocation::descriptor<T> get_allocation(const T *ptr) const;

  /// Instruct buffer to free the allocation on the specified device at buffer
  /// destruction.
  /// Throws \c invalid_parameter_error if no allocation for specified device
  /// exists.
  void own_allocation(const device &dev);

  /// Instruct buffer to free the allocation at buffer destruction.
  /// \c ptr must be an existing allocation managed by the buffer.
  /// If \c ptr cannot be found among the managed memory allocations,
  /// \c invalid_parameter_error is thrown.
  void own_allocation(const T *ptr);

  /// Instruct buffer to not free the allocation on the specified device
  /// at buffer destruction.
  /// Throws \c invalid_parameter_error if no allocation for specified device
  /// exists.
  void disown_allocation(const device &dev);

  /// Instruct buffer to not free the allocation associated with the provided
  /// pointer.
  /// Throws \c invalid_parameter_error if no allocation managed by the buffer
  /// is described by \c ptr.
  void disown_allocation(const T *ptr);

}; // class buffer

} // namespace sycl

```

## Constructing buffer objects on top of USM pointers

AdaptiveCpp can also construct buffers on top of existing USM pointers.

### API reference

```c++

namespace sycl {
namespace buffer_allocation {

/// A type based on buffer_allocation::descriptor<T> that stores additionally
/// information about the data state (outdated/current) of the data referenced
/// by the descriptor
template<class T>
using tracked_descriptor = __unspecified__;

enum class management_mode {
  owning,
  non_owning
};

static constexpr bool take_ownership = management_mode::owning;
static constexpr bool no_ownership = management_mode::non_owning;

/// Construct descriptor for a data region with initially outdated content.
/// When first accessed, the runtime will emit data transfers to update
/// date content.
tracked_descriptor<T> empty_view(T *ptr, device dev,
                                 management_mode m = no_ownership);

/// Construct descriptor for a data region with initially recent content.
/// When first accessed, the runtime will not emit data transfers to
/// update data content. Instead, the allocation may be used as a
/// potential data source to update other outdated allocations from.
tracked_descriptor<T> view(T *ptr, device dev, management_mode m = no_ownership);

}
}

template <typename T, int dimensions,
          typename AllocatorT>
class buffer {
public:
  /// Construct buffer on top of existing USM pointers with given range.
  /// Will not write back at destruction unless set_final_data() is used.
  /// Modifying the USM pointers externally during the duration of the buffer lifetime
  /// should be avoided unless the user has expert knowledge of AdaptiveCpp's data
  /// state tracking mechanisms.
  buffer(
    const std::vector<buffer_allocation::tracked_descriptor<T>>& input_allocations,
    const range<dimensions>& r,
    const property_list& prop_list = {});
  
  buffer(
    const std::vector<buffer_allocation::tracked_descriptor<T>>& input_allocations,
    const range<dimensions>& r,
    AllocatorT allocator,
    const property_list& prop_list = {});
};

```

## Example code

```c++

sycl::queue q;
sycl::range size{1024};

int* alloc1 = sycl::malloc_shared<int>(size.size(), q);
int* alloc2 = sycl::malloc_shared<int>(size.size(), q);

{
  // Construct buffer on top of alloc1. Use empty_view() because alloc1
  // does not old live data.
  // Note: To give ownership over the pointer to the pointer, pass
  // the optional argument take_ownership to empty_view()
  sycl::buffer<int> b1{
      {sycl::buffer_allocation::empty_view(alloc1, q.get_device())}, size};

  // The buffer now uses the given allocation for all operations on
  // q.get_device()
  assert(b1.has_allocation(q.get_device()));
  assert(b1.get_pointer(q.get_device()) == alloc1);

  // Iterate over all allocations of the buffer. We did not grant
  // ownership of the pointer to the buffer. This is checked here.
  b1.for_each_allocation([&](const auto& alloc){
    if(alloc.ptr == alloc1){
      assert(!alloc.is_owned);
    }
  });

  // We can use the buffer just as usual
  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> acc{b1, cgh};

    cgh.parallel_for(size, [=](sycl::id<1> idx){
      acc[idx] = idx.get(0);
    });
  });
}

{
  // Construct another buffer - this time we use view instead of empty_view
  // because the pointer now holds up-to-date data
  sycl::buffer<int> b2{
      {sycl::buffer_allocation::view(alloc1, q.get_device())}, size};
  
  // Can again use the buffer just as usual
  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> acc{b2, cgh};

    cgh.parallel_for(size, [=](sycl::id<1> idx){
      alloc2[idx.get(0)] = acc[idx];
    });
  });

  sycl::host_accessor<int> hacc{b2};
  for(int i = 0; i < size.get(0); ++i){
    assert(hacc[i] == i);
  }  
}

sycl::free(alloc1, q);
sycl::free(alloc2, q);

```