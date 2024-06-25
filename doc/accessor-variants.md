# AdaptiveCpp accessor variants

## Terminology

In the following, we will use the following terminology:
* A *placeholder* accessor is an accessor that was constructed without providing a SYCL `handler` object. Placeholder need to be bound to a `handler` by calling `handler::require()` at a later point.
* A *ranged* accessor is an accessor that is constructed by specifying an access range consisting of a `sycl::range` and optionally a `sycl::id` describing the access offset. Ranged accessors may be used if only a subset of the buffer is accessed.
* An *unranged* accessor is an accessor that is not a ranged accessor.

## Motivation

SYCL accessors need to store several different kinds of information, depending on their use case. In general, this includes:
* A data pointer
* The n-dimensional shape of the allocation
* the access offset and access range
* Certain flags, e.g. whether the accessor is `no_init`
* Placeholder accessors generally need to store a `shared_ptr` or similar to the `buffer` object
* Additional implementation-specific information that the SYCL runtime needs to enable core accessor functionality.

As a consequence, accessors are, in general, not light-weight objects. Accessor bloat however is highly undesirable, because it can have significant negative performance impact in kernels dominated by register pressure. This is especially true since accessors are a primary mechanism to operate on data, and therefore tend to be used in the most computationally intensive parts of the kernel.

In a compiler-based SYCL implementation, some of the information stored inside accessors can be stripped if it is not used. However, in a SYCL implementation that tries to minimize compiler engineering efforts such as AdaptiveCpp, or even an entirely library-based implementation, this is very difficult or even impossible.

But even in compiler-based SYCL implementations, information may be duplicated unnecessarily. Consider the following example:

```c++
sycl::buffer<int> b1{1024};
sycl::buffer<int> b2{1024};

sycl::queue q;
q.submit([&](sycl::handler& cgh){
  sycl::accessor<int> acc1 {b1, cgh, sycl::no_init};
  sycl::accessor<int> acc2 {b2, cgh, sycl::no_init};

  cgh.parallel_for(sycl::range{1024}, [=](sycl::id<1> idx){
    
  });
});
```
Here, the the range of the buffer for both `acc1` and `acc2` will have to be passed into the kernel, because the compiler cannot know in general that the buffers have the same size. After all, the buffer size is a runtime parameter and in general unknown at compile-time. Consequently, the range of the accessors will be duplicated for both `acc1` and `acc2`, and take up unnecessary space in the kernel. In standard SYCL 2020, there is no way for the user to enforce that the accessor sizes are, in fact, equal and hence only need to be stored once.

## AdaptiveCpp accessor variants

The AdaptiveCpp accessor variants extension is based on the realization that accessor size can be optimized if the way in which the accessor was constructed enters the accessor type, e.g. via template parameters.

For example, an unranged accessor does not have to store access offset and an additional access range - it only needs to store the buffer allocation shape.

With SYCL 2020 deduction guides and C++17 class template argument deduction, these additional template parameters can even be deduced automatically.

AdaptiveCpp introduces the following accessor variants, which are described by the `sycl::accessor_variant` enumeration:
* `ranged_placeholder` - ranged placeholder accessor.
* `unranged_placeholder` - unranged placeholder accessor
* `ranged` - ranged non-placeholder accessor
* `unranged` - unranged non-placeholder accessor
* `raw` - A minimal light-weight accessor that effectively only stores a pointer, at the expense that it cannot expose certain parts of the regular accessor API. See the *Restrictions* section below for more details.

The following table describes which data is stored in the different accessor variants.

| | `raw` | `unranged` | `ranged` | `unranged_placeholder` | `ranged_placeholder` |
|------------------|------------------|------------------|------------------|------------------|------------------|
| Core runtime information | Yes | Yes | Yes | Yes | Yes |
| Data pointer | Yes | Yes | Yes | Yes | Yes |
| buffer range | No | Yes | Yes | Yes | Yes |
| access offset | No [1] | No [1] | Yes | No [1] | Yes |
| access range  | No | No [2] | Yes | No [2] | Yes |
| `shared_ptr` to `buffer` | No | No | No | Yes | Yes |
| Accessor properties (e.g. `no_init`) | No | No | No | Yes | Yes |

* [1] Queries for offset return zero-initialized `sycl::id<>` object
* [2] Queries for access range return buffer range

In order to avoid incompatibility with SYCL 2020 and SYCL 1.2.1 as much as possible, AdaptiveCpp accessor variants are exposed by extending the existing `access::placeholder` accessor template parameter with additional values for the accessor variants:

```c++
namespace sycl {

enum class accessor_variant {
  false_t, // compatibility with SYCL 1.2.1 placeholder enum
  true_t,  // compatibility with SYCL 1.2.1 placeholder enum
  ranged_placeholder,
  ranged,
  unranged_placeholder,
  unranged,
  raw
};

namespace access {
// Deprecated (reused in AdaptiveCpp to store accessor variants)
using placeholder = sycl::accessor_variant;
} // access
} // sycl

template <typename dataT, int dimensions,
          access_mode accessmode,
          target accessTarget,
          accessor_variant AccessorVariant>
class accessor { ... };
```

However, note that using distinct accessor variants might potentially break certain code patterns. While common accessor usage works, the introduction of new distinct accessor types can break the assumption of compatibility between different accessors. For example, you might not be able to store different accessors in a container anymore.

## Constructing AdaptiveCpp accessor variants

hipSYCL accessor variants can be constructed in the following way:
1. By explicitly setting the `accessor_variant` template parameter of the accessor to a value that differs from the standard `access::placeholder::false_t` and `access::placeholder::true_t`.
2. By using the `sycl::raw_accessor`, `sycl::ranged_accessor`, `sycl::unranged_accessor`, `sycl::ranged_placeholder_accessor`, `sycl::unranged_placeholder_accessor` type aliases (see API reference below)
3. For raw accessors, by using the new `read_only_raw`, `read_write_raw` and `write_only_raw` deduction tags
4. If `ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION` is defined, SYCL 2020 CTAD rules and `buffer::get_access()` will automatically construct accessors of the most efficient types.

Example code:
```c++

sycl::buffer<int> buff{size};
// If ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION is enabled,
// constructs accessor_variant::unranged_placeholder
sycl::accessor unranged_placeholder{buff, sycl::read_write, sycl::no_init};
// If ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION is enabled,
// constructs accessor_variant::ranged_placeholder
sycl::accessor ranged_placeholder{buff, subrange, offset, sycl::read_write,
                                    sycl::no_init};


q.submit([&](sycl::handler &cgh) {
  // If ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION is enabled,
  // constructs accessor_variant::unranged
  sycl::accessor unranged_acc{buff, cgh, sycl::read_write, sycl::no_init};
  // If ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION is enabled,
  // constructs accessor_variant::ranged
  sycl::accessor ranged_acc{
      buff, cgh, subrange, offset, sycl::read_write, sycl::no_init};

  // Construct raw accessor
  sycl::accessor raw_acc{buff, cgh, sycl::read_write_raw, sycl::no_init};
  // Construct raw accessor
  sycl::raw_accessor<int> raw_acc2{buff, cgh, sycl::no_init}

  auto kernel = [=](sycl::id<1> idx){
    // operator[] for raw accessors is only available for 0D or 1D
    // raw accessors. We are in 1D here, so this is fine.
    raw_acc[idx] = idx.get(0);
  };

  cgh.parallel_for(size, kernel);
});
```

## Restrictions

There are certain restrictions with respect to the functionality of individual accessor variants and how they can be converted into each other.

### Conversion rules
Two accessor objects of different accessor variant can be implicitly converted, unless one of the following conditions is met:
* the source accessor is a raw accessor. In that case the destination accessor would expose more information than the source could provide.
* the destination accessor is unranged for a ranged source accessor, or a standard SYCL 2020 accessor. In that case, the destination accessor might expose access to regions of the buffer that are not guaranteed to be in a consistent state on the target device, if they are outside of the original access range of the source accessor.
* the destination accessor is a placeholder accessor for a non-placeholder source accessor.

Conversion from AdaptiveCpp accessor variants to standard accessors (variant is `access::placeholder::false_t` or `access::placeholder::true_t`) is supported.
The opposite direction is currently allowed for compatibility reasons, but might be disabled in the future as correctness for ranged accessors cannot be preserved.

### Raw accessors

Raw accessors additionally obey the following restrictions:
* They cannot be constructed as placeholders
* They do not expose range and size queries
* They only expose `operator[]` for 0 and 1-dimensional accessors
* The raw accessor `operator[]`, if available, does not take into account any access offset.

## API reference

```c++
namespace sycl {

enum class accessor_variant {
  false_t, // compatibility with SYCL 1.2.1 placeholder enum
  true_t,  // compatibility with SYCL 1.2.1 placeholder enum
  ranged_placeholder,
  ranged,
  unranged_placeholder,
  unranged,
  raw
};

namespace access {
// Deprecated (reused in hipSYCL to store accessor variants)
using placeholder = sycl::accessor_variant;
} // access

// New deduction tags for raw accessors
inline constexpr __unspecified__ read_only_raw;
inline constexpr __unspecified__ read_write_raw;
inline constexpr __unspecified__ write_only_raw;

// Note: This synopsis of the accessor class is incomplete
// and focuses on the aspects modified by this extension.
template <typename dataT, int dimensions,
          access_mode accessmode,
          target accessTarget,
          accessor_variant AccessorVariant>
class accessor
{
public:
  // Only available for 0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           const property_list &prop_list = {});

  // Only available for 0D accessors
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           const property_list &prop_list = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef, TagT tag,
           const property_list &prop_list = {});

  // Only available for >0D accessors
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {});

  // Only available for >0D accessors
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, TagT tag,
           const property_list &prop_list = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, TagT tag,
           const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset,
           const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // raw, ranged and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset, TagT tag,
           const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // unranged_placeholder and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // unranged_placeholder and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           TagT tag, const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // unranged_placeholder and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, const property_list &propList = {});

  // Only available for >0D accessors; not available for
  // unranged_placeholder and unranged accessor variants
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, TagT tag,
           const property_list &propList = {});

  // Copy-constructor is guaranteed to be trivial for copies
  // between the same accessor variant
  accessor(const accessor& other) = default;

  // Assignment operator is guaranteed to be trivial for copies
  // between the same accessor variant
  accessor& operator=(const accessor& other) = default;

  // Only available for read-only accessors.
  // Not available if, for a source accessor variant S 
  // and a destination accessor variant D:
  // * S is a raw accessor
  // * S is ranged and D is unranged
  // * D is a placeholder and S is not a placeholder
  template <access::placeholder P>
  accessor(const accessor<std::remove_const_t<dataT>, dimensions,
                          access_mode::read_write, accessTarget, P> &other);

  // Conversion between different accessor variants
  // Not available if, for a source accessor variant S 
  // and a destination accessor variant D:
  // * S is a raw accessor
  // * S is ranged and D is unranged
  // * D is a placeholder and S is not a placeholder
  template <accessor_variant OtherV>
  accessor(const accessor<dataT, dimensions, access_mode::read_write,
                          accessTarget, OtherV> &other);

  
  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  size_t get_size() const noexcept;

  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  size_t get_count() const noexcept;

  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  size_t get_count() const noexcept;
  
  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  size_t byte_size() const noexcept;
  }

  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  size_t size() const noexcept;

  // Only available if the accessor is not a raw accessor, or
  // the dimensionality is 0.
  bool empty() const noexcept;

  // Only available if the accessor is not a raw accessor
  range<dimensions> get_range() const noexcept;

  // Only available when dimensions > 0 
  id<dimensions> get_offset() const noexcept;
  
  // Only available for non-atomic accessors that either
  // * are not raw accessors
  // * are raw accessors of dimensionality <= 1. In this case,
  //   the access offset will not be taken into account.
  reference operator[](id<dimensions> index) const noexcept;
  
  // Only available for non-atomic accessors of dimensionality 1 
  // that either
  // * are not raw accessors
  // * are raw accessors of dimensionality <= 1. In this case,
  //   the access offset will not be taken into account.
  reference operator[](size_t index) const noexcept;

  // Only available for atomic accessors that either
  // * are not raw accessors
  // * are raw accessors of dimensionality <= 1. In this case,
  //   the access offset will not be taken into account.
  atomic<dataT, access::address_space::global_space>
  operator[](id<dimensions> index) const noexcept;

  // Only available for atomic accessors that either
  // * are not raw accessors
  // * are raw accessors of dimensionality <= 1. In this case,
  //   the access offset will not be taken into account.
  atomic<dataT, access::address_space::global_space>
  operator[](size_t index) const noexcept;

  // Only available for accessors of dimensionality > 1 that
  // are not raw accessors
  __unspecified__ operator[](size_t index) const noexcept;

};


template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using raw_accessor = accessor<T, Dim, M, Tgt, accessor_variant::raw>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using ranged_accessor = accessor<T, Dim, M, Tgt, accessor_variant::ranged>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using unranged_accessor = accessor<T, Dim, M, Tgt, accessor_variant::unranged>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using ranged_placeholder_accessor =
    accessor<T, Dim, M, Tgt, accessor_variant::ranged_placeholder>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using unranged_placeholder_accessor =
    accessor<T, Dim, M, Tgt, accessor_variant::unranged_placeholder>;


// Deduction guides when ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION is active

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, TagT tag,
         const property_list &prop_list = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::unranged_placeholder>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         TagT tag, const property_list &prop_list = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::unranged>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::ranged_placeholder>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::ranged_placeholder>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, TagT tag, const property_list &propList = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::ranged>;

template <typename T, int Dim, typename AllocatorT, typename TagT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, id<Dim> accessOffset, TagT tag,
         const property_list &propList = {})
    -> accessor<T, Dim, /* deduced mode */,
                /* deduced target */,
                accessor_variant::ranged>;


}
```