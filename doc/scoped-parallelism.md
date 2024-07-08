# Scoped parallelism (Version 2)

## Introduction

Scoped parallelism provides a novel way to formulate kernels in a hierarchical and performance-portable way. It contributes the following features to the SYCL world:
* Hierarchical kernels with arbitrary nesting depth, while avoid the implementation and performance pitfalls from old SYCL 1.2.1 hierarchical parallelism by giving the implementation more freedom;
* Performance portability (`nd_range` parallel for is notoriously difficult to implement efficiently in library-only CPU implementations);
* Multi-dimensional (potentially nested) subgroups;
* group algorithms in another model apart from `nd_range` parallel for;
* A model that allows for compile-time hints encoded in the group type that can be used by e.g. group algorithms to provide optimized implementations;
* A hierarchical-like model that is built using SYCL 2020 instead of 1.2.1 concepts, such as multiple group types.

Scoped parallelism expands hierarchical parallelism concepts from SYCL 1.2.1 by introducing a group hierarchy below work group scope, including subgroups, and potentially further group types such as sub-subgroups depending on the backend. This can allow the backend to automatically use tiling optimization strategies. It also solves performance and implementation issues in traditional SYCL 1.2.1 hierarchical parallelism by giving the SYCL implementation more freedom with respect to the degree of parallelism that is employed in executing code:
Scoped parallelism acknowledges that an implementation may achieve better performance by employing a different degree of parallelization within a work group compared to the user-requested work group size.

In scoped parallelism, the backend chooses a number of parallel execution resources (e.g. threads on a CPU, OpenCL work items, CUDA threads etc.) to execute the kernel within each work group. We refer to this actual parallelism utilized by the backend as *physical parallelism*. Each work group is executed by a number of *physical work items* that are part of a *physical group size*. The user can query the position within and size of the physical parallel iteration space at any time.
A user-provided logical iteration space is then distributed across the physical one. Optionally, the logical iteration space can be subdivided hierarchically into smaller units.
This allows to formulate the program in a hierarchical manner.

Structurally, within a work group, scoped parallelism is similar to patterns found in other programming models like OpenMP, where a (by default) implementation-defined number of threads is first spawned with `#pragma omp parallel` and later on, a user-provided parallel iteration space is distributed across the physical one using `#pragma omp for`.
Scoped parallelism works analogously: `handler::parallel()` launches a parallel kernel, and `sycl::distribute_items()` distributes the user-provided iteration space across the execution resources.

Because scoped parallelism enforces explicit information about where to allocate variables from the user, performance surprises as in hierarchical parallel for are prevented. In particular, simple variable declarations will not be allocated in local memory (as in hierarchical parallelism), but in the private memory of the physical work item which will always be efficient.

## `distribute_items()`

When submitting a kernel using `handler::parallel()` or `queue::parallel()`, the user provides a *logical group size* that describes the number of logical work items. When calling the `sycl::distribute_items()` function, the backend will distribute the items from the logical iteration space of the provided group across the available physical work items that have been assigned to processing the group.

```c++
sycl::range<1> num_work_groups = ...
sycl::range<1> logical_group_size = ...

// Submit num_work_groups groups with a given logical size each
sycl::queue{}.parallel(num_work_groups, logical_group_size,
  // The group is provided as an argument to the user-provided kernel
  [=](auto group){
    // Code here will be executed potentially multiple times in parallel,
    // depending on the physical parallelism.
    // Variables declared outside of distribute_items() will be allocated
    // in the private memory of the physical work item.
    sycl::distribute_items(group, [&](sycl::s_item<1> logical_idx){
      // Code here will be executed once per logical item in the logical
      // iteration space.
    });
  });
```

In general, the backend will configure scoped parallelism groups with compile-time optimization hints that enter the type as template arguments. It is therefore strongly recommended to use `auto` or template parameters to accept any group type inside kernels. To be more specific, the provided top-level work group is of type `sycl::s_group<class __UnspecifiedParameter__>`. While the precise name of this type might change in the future, it is guaranteed that this type is *not* the same as the regular `sycl::group`.

## `distribute_groups()`

Additionally, the user can invoke `sycl::distribute_groups()`, which instructs the backend that the user would like to subdivide a group into smaller units (this can be useful for tiling optimization strategies). `distribute_groups()` will then attempt to provide subdivided groups to the user, and distribute a number of physical work items across the new groups.

`distribute_groups()` invocations can be nested arbitrarily deep. However, the size of the provided smaller groups is backend-defined, and might also depend on the device, specific kernels, or kernel parameters. For example, if the logical work group size is not divisible by sub group sizes that can be executed by the backend or device, the SYCL implementation might be forced to instead subdivide into trivial scalar groups that only contain a single work item.

*Note:* After a sufficiently deep nesting level, only scalar groups will be returned from then on, but the nesting level where this happens is unspecified.

When invoking `distribute_groups`, a number of physical work items from the parent group is assigned to each subdivided group. Inside the subdivided groups, `distribute_items` can again be used to then distribute the logical iteration space across the available physical work items.

```c++
sycl::range<1> num_work_groups = ...
sycl::range<1> logical_group_size = ...

sycl::queue{}.parallel(num_work_groups, logical_group_size,
  [=](auto group){
    // Note that the group argument is of generic auto type;
    // this allows the implementation to provide arbitrary group
    // types that are optimized for the backend.
    sycl::distribute_groups(group, [&](auto subgroup){
      sycl::distribute_groups(subgroup, [&](auto subsubgroup){
        sycl::distribute_groups(subsubgroup, [&](auto subsubsubgroup){
          // distribute_items() to make sure code is executed for each logical item
          sycl::distribute_items(subsubsubgroup, [&](sycl::s_item<1> logical_idx){
            ...
          })
        });
      });
    });
  });
```

There are three different group categories in scoped parallelism. They can be distinguished using the `static constexpr sycl::memory_scope fence_scope` member.
1. The work group. For this group, it holds that `Group::fence_scope == sycl::memory_scope::work_group`. Represents the entire work group.
2. A subgroup. For subgroups it holds that `Group::fence_scope == sycl::memory_scope::sub_group`. A sub group is a collection of logical work items below the granularity of the full work group. A backend or device might support multiple levels of nested sub groups.
3. A scalar group. For scalar groups it holds that `Group::fence_scope == sycl::memory_scope::work_item`. Scalar groups contain a single logical work item.

 
Invoking `distribute_groups()` on a group may result in the following group types provided in the next nesting level:
1. Work group subdivision may result in subgroup or a scalar group.
2. Subdivision of a subgroup may result in another, smaller subgroup or a scalar group
3. Subdivision of a scalar group will result in a scalar group.

`sycl::distribute_items()` within 
The user then expresses the kernel using calls to `distribute_groups` and `distribute_items`. `distribute_groups`

## Scoped parallelism nesting rules

Any program not obeying any of the following rules is **illegal**.
1. The group object passed to `distribute_items(), distribute_groups(), single_item()`, their `*_and_wait` counterparts as well as group algorithms (e.g. `group_barrier()`) must be the smallest available group subunit available at that point in the code. In other words, it must be the group object that is "closest" in the nesting hierarchy. Otherwise the code would request execution within a parent group from a subunit of the parent group, at which point not all items from the parent group might participate anymore.
2. `distribute_items(), distribute_groups(), single_item()`, their `*_and_wait` counterparts as well as group algorithms (e.g. `group_barrier()`) may not be called from within `distribute_items()` calls. `s_private_memory` objects may not be declared inside `distribute_items()` calls.
3. `distribute_items(), distribute_groups(), single_item()`, their `*_and_wait` counterparts as well as group algorithms (e.g. `group_barrier()`) are collective with respect to the physical work items of the group object they operate on. The user must guarantee that each physical work item of the provided group object reaches their callsite. As soon as one physical item of a group invokes those functions, all others from the same group need to invoke it as well. There are no requirements regarding other, independent groups.

```c++
//Example of okay:
sycl::queue{}.parallel(num_work_groups, logical_group_size,
  [=](auto group){

    sycl::distribute_items(group, [&](sycl::s_item<1> idx){
      ...
    });
    sycl::group_barrier(group);
    sycl::distribute_groups(group, [&](auto subgroup){
      sycl::distribute_items(subgroup, [&](sycl::s_item<1> idx){
        ...
      });
      sycl::single_item(subgroup, [&](){
        ...
      });
      sycl::distribute_groups(subgroup, [&](auto subsubgroup){
        sycl::distribute_items(subsubgroup, [&](sycl::s_item<1> idx){
          ...
        });
      });
      sycl::group_barrier(subgroup);
    });
  });

// Example of NOT okay
sycl::queue{}.parallel(num_work_groups, logical_group_size,
  [=](auto group){

    sycl::distribute_items(group, [&](sycl::s_item<1> idx){
      // Invalid (rule 2): Group algorithm from within distribute_items()
      sycl::group_barrier(group);
    });
    sycl::distribute_groups(group, [&](auto subgroup){
      // Invalid (rule 1): distribute_items() is not operating on
      // the closest group object (should operate on subgroup)
      sycl::distribute_items(group, [&](sycl::s_item<1> idx){
        // Invalid (rule 2): single_item() is invoked from within
        // distribute_items()
        sycl::single_item(subgroup, [&](){
          ...
        });
      });
      
      sycl::distribute_groups(subgroup, [&](auto subsubgroup){
        // Invalid (rule 3): Only one physical work item (the leader)
        // of subsubgroup will reach the distribute_items call.
        if(subsubgroup.leader()){
          sycl::distribute_items(subsubgroup, [&](sycl::s_item<1> idx){
            ...
          });
        }
      });
      // Invalid (rule 1): group_barrier() is not operating on
      // the closest group object (should operate on subgroup)
      sycl::group_barrier(group);
    });
  });
```

## Synchronization

There are two ways of synchronizing:
* `distribute_items_and_wait()`, `distribute_groups_and_wait()`, `single_item_and_wait()` automatically insert a barrier on the group provided as argument to those functions after executing the user-provided callable.
* alternatively, `group_barrier()` can be invoked directly outside of `distribute_items` and in the appropriate scope of the group. This can provide more control as the fence scope can also be specified in this way.

`distribute_items()`, `distribute_groups()`, `single_item()` do not synchronize.

## Memory placement rules

In the scoped parallelism model, the following memory placement rules apply:
* Variables declared inside a `distribute_items()` call will be allocated in private memory of the logical work item.
* Variables declared outside of `distribute_items()` calls will be allocated in the private memory of the executing physical work item.
* Variables can be explicitly but into either local memory or the private memory of logical work items using `sycl::memory_environment()`

### Explicit allocation of local and private memory

`sycl::memory_environment()` can be used to explicitly allocate either local or private memory. To this end, an arbitrary number of memory allocation requests can be passed to `memory_environment()`. For each provided memory allocation request, a reference to the requested memory will then be passed into the last argument of `memory_environment`: A callable that accepts references to the memory.

Memory allocation requests of type T can be formulated using `sycl::require_local_mem<T>()` and `sycl::require_private_mem<T>()`. Optionally, these functions can accept an argument that will be used to initialize the memory content of the allocation. See the API reference for more details.
Note that while for `sycl::require_local_mem<T>()` the allocation will be passed into the callable as type `T&`, for `sycl::require_private_mem<T>()` a reference to a wrapper object will be passed into the callable instead. The private memory allocation can then be accessed by passing the `sycl::s_item` object to its `operator()`.
This is necessary because `sycl::require_private_mem<T>()` is a request to allocate in the private memory of the *logical* work item, which however may not exist outside of `distribute_items()`.

See the example code below for an illustration on how `memory_environment()` can be used.

## Example code

```c++
#include <sycl/sycl.hpp>
#include <vector>

int main(){
  
  sycl::queue q;
  
  std::size_t input_size = 1024;
  std::vector<int> input(input_size);
  for(int i = 0; i < input.size(); ++i)
    input[i] = i;
  
  sycl::buffer<int> buff{input.data(), sycl::range<1>{input_size}};
  
  constexpr size_t Group_size = 128;
  q.submit([&](sycl::handler& cgh){
    auto data_accessor = buff.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel<class Kernel>(sycl::range<1>{input_size / Group_size}, sycl::range<1>{Group_size}, 
    [=](auto grp){
      // Outside of distribute_items(), the degree of parallelism is implementation-defined.
      // the implementation can use whatever is most efficient for hardware/backend.
      // On AdaptiveCpp CPU backend, this would be executed by a single thread on CPU
      // and Group_size threads on AdaptiveCpp GPU backend.
      // Information about the position in the physical iteration space can be obtained
      // using grp.get_physical_local_id() and grp.get_physical_local_range().

      // sycl::memory_environment() can be used to allocate local memory 
      // (of compile-time size) as well as private memory that is persistent across
      // multiple distribute_items() calls.
      // Of course, local accessors can also be used.
      sycl::memory_environment(grp, 
        sycl::require_local_mem<int[Group_size]>(),
        // the requested private memory is not used in this example,
        // and only here to showcase how to request private memory.
        sycl::require_private_mem<int>(),
        [&](auto& scratch, auto& private_mem){
        // the arguments passed to the lambda function corresponds to the require arguments before.
        // private_mem is allocated in private memory of the _logical_ work item
        // and of type sycl::s_private_memory<T, decltype(grp)>&.
        // scratch is a reference to the requested int[Group_size] array.

        // Variables not explicitly requested as local or private memory 
        // will be allocated in private memory of the _physical_ work item
        // (see the for loop below)

        // `distribute_items` distributes the logical, user-provided iteration space across the physical one. 
        sycl::distribute_items(grp, [&](sycl::s_item<1> idx){
            scratch[idx.get_local_id(grp, 0)] = data_accessor[idx.get_global_id(0)]; 
        });
        // Instead of an explicit group_barrier, we could also use the
        // blocking distribute_items_and_wait()
        sycl::group_barrier(grp);

        // Can execute code e.g. for a single item of a subgroup:
        sycl::distribute_groups(grp, [&](auto subgroup){
          sycl::single_item(subgroup, [&](){
            // ...
          });
        });

        // Variables inside the parallel scope that are not explicitly local or private memory
        // are allowed, if they are not modified from inside `distribute_items()` scope.
        // The SYCL implementation will allocate those in private memory of the physical item,
        // so they will always be efficient. This implies that the user should not attempt to assign values
        // per logical work item, since they are allocated per physical item.
        for(int i = Group_size / 2; i > 0; i /= 2){
          // The *_and_wait variants of distribute_groups and distribute_items
          // invoke a group_barrier at the end.
          sycl::distribute_items_and_wait(grp, 
            [&](sycl::s_item<1> idx){
            size_t lid = idx.get_innermost_local_id(0);
            if(lid < i)
              scratch[lid] += scratch[lid+i];
          });
        }
        
        sycl::single_item(grp, [&](){
          data_accessor[grp.get_group_id(0)*Group_size] = scratch[0];
        });
      });
    });
  });
  
  // Verify results on host
  auto host_acc = buff.get_access<sycl::access::mode::read>();
  
  for(int grp = 0; grp < input_size/Group_size; ++grp){
    int host_result = 0;
    for(int i = grp * Group_size; i < (grp+1) * Group_size; ++i)
      host_result += i;
    
    if(host_result != host_acc[grp*Group_size])
      std::cout << "Wrong result, got " << host_acc[grp * Group_size] << ", expected " << host_result << std::endl;
  }
}
```

## API reference

### New functions in `sycl::handler`

```c++
/// Spawns a new parallel region. f must have the signature 
/// void(sycl::s_group<__unspecified__>)
template <typename KernelName, typename KernelFunctionType,
          typename... Reductions, int dimensions>
void handler::parallel(range<dimensions> numWorkGroups,
                      range<dimensions> workGroupSize,
                      Reductions... redu,
                      const KernelFunctionType& f);
```


## New free functions

```c++

namespace sycl {

/// Only available if SpGroup is a scoped parallelism group type.
///
/// Distributes the logical iteration space within the provided group
/// across the physical iteration space of the group, i.e. the
/// physical parallel resources that participate in executing this group.
///
/// For each logical work item within the group, executes f exactly once.
/// 
/// f must be a callable with signature void(sycl::s_item<Dimension>)
template<class SpGroup, class NestedF>
void distribute_items(const SpGroup &g, NestedF f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type.
///
/// Equivalent to: distribute_items(g, f); group_barrier(g);
template<class SpGroup, class NestedF>
void distribute_items_and_wait(const SpGroup &g, NestedF f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type
///
/// Subdivides the logical iteration space of the group into 
/// implementation-defined smaller units and distributes
/// them across the physical parallel resources that participate in execution.
///
/// f is executed once per subgroup. f must be a callable of signature
/// void(__ImplementationDefinedScopedParallelismGroupType__)
template <class SpGroup, class NestedF>
void distribute_groups(const SpGroup &g, NestedF f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type
///
/// Equivalent to distribute_groups(g, f); group_barrier(g);
template <class SpGroup, class NestedF>
void distribute_groups_and_wait(const SpGroup &g, NestedF f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type
///
/// Ensures that f() is executed exactly once within the group g.
/// Must not be used within distribute_items().
///
/// f must be a callable of signature void().
template <class SpGroup, class F>
void single_item(const SpGroup &g, F f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type
///
/// Equivalent to single_item(g, f); group_barrier(g);
template <class SpGroup, class F>
void single_item_and_wait(const SpGroup &g, F f) noexcept;

/// Only available if SpGroup is a scoped parallelism group type
///
/// Constructs a memory environment for the given group. Currently,
/// Only top-level work groups are supported. 
///
/// Args is a collection of memory allocation requests (return values of 
/// sycl::require_local_memory() and sycl::require_private_memory()) and, in
/// the last argument, a callable.
///
/// This callable will be invoked once the required memory has been allocated.
/// The callable will be provided with a references to the memory allocations, 
/// in the order they were requested.
///
/// If private memory was requested, a wrapper type will be passed into the callable
/// instead of the raw memory reference. The wrapper type has a member function 
/// T& operator()(const s_item<Dim>&) that can be used to obtain the actual memory.
/// Local memory allocations will be passed directly as reference into the callable.
template <class SpGroup, typename... Args>
void memory_environment(const Group &g, Args&&... args) noexcept;

/// Synonym for memory_environment(g, require_local_mem<T>(), f)
template<class T, class Group, class Function>
void local_memory_environment(const Group& g, Function&& f);

/// Synonym for memory_environment(g, require_private_mem<T>(), f)
template<class T, class Group, class Function>
void private_memory_environment(const Group& g, Function&& f) noexcept;

/// Construct a request for memory_environment() to allocate local memory
/// of the provided type T. The memory will not be initialized.
template <class T>
__unspecified__ require_local_mem() noexcept;

/// Construct a request for memory_environment() to allocate local memory
/// of the provided type T.
///
/// If T is of a C-array type with up to 3 dimensions, i.e. can be represented
/// as ScalarT [N], ScalarT [N][M] or ScalarT [N][M][K], the argument x must be of
/// type ScalarT. Each array element will be initialized with the value of x.
///
/// Otherwise, InitType must be the same as T, and the allocation will be initialized
/// with the value of x.
template <class T, class InitType>
__unspecified__ require_local_mem(const InitType& x) noexcept;

/// Construct a request for memory_environment() to allocate private memory
/// of the provided type T. The memory will not be initialized.
///
/// The callable passed to memory_environment() will not be provided with
/// the requested private memory allocation directly, but with a wrapper object.
/// A reference to the actual allocation can be obtained by passing an s_item object
/// to the operator() of the wrapper object.
template <class T>
__unspecified__ require_private_mem() noexcept;

/// Construct a request for memory_environment() to allocate private memory
/// of the provided type T. The memory will be initialized using the value of
/// x.
///
/// The callable passed to memory_environment() will not be provided with
/// the requested private memory allocation directly, but with a wrapper object.
/// A reference to the actual allocation can be obtained by passing an s_item object
/// to the operator() of the wrapper object.
template <class T>
__unspecified__ require_private_mem(const T& x) noexcept;

}

```

### New Scoped group concept

All scoped parallelism groups obey the new scoped group concept, which expands the regular SYCL 2020 group concept.

```c++
// Concept that is implemented by multiple classes at group, subgroup and
// work item scope.
template<class Properties>
class ScopedGroup {

  static constexpr int dimensions = /* dimensionality of the group */
  using id_type = sycl::id<dimensions>;
  using range_type = sycl::range<dimensions>;
  using linear_id_type = /* uint32_t or uint64_t */
  
  /// The memory scope the group operates in. 
  /// This can be used to distinguish different group types.
  static constexpr memory_scope fence_scope = /* memory scope of the group */;

  /// Returns the id of the group within the parent group.
  /// If the group is a work group, there is no parent group
  /// and the work group id is returned.
  id_type get_group_id() const noexcept;

  /// Returns the id of the group within the parent group
  /// for the specified dimension.
  /// If the group is a work group, there is no parent group
  /// and the work group id is returned.
  size_t get_group_id(int dimension) const noexcept;

  /// Returns the linear id of the group within the parent group.
  /// If the group is a work group, there is no parent group
  /// and the work group id is returned.
  linear_id_type get_group_linear_id() const noexcept;

  /// Returns the number of groups within the parent group.
  /// If this group is a work group, there is no parent group
  /// and the number of work groups is returned.
  range_type get_group_range() const noexcept;
  
  /// Returns the number of groups within the parent group
  /// in the specified dimension.
  /// If this group is a work group, there is no parent group
  /// and the number of work groups is returned.
  size_t get_group_range(int dimension) const noexcept;

  /// Returns the number of groups within the parent group.
  /// If this group is a work group, there is no parent group
  /// and the number of work groups is returned.
  size_t get_group_linear_range() const noexcept;

  /// Equivalent to get_group_id(dimension)
  size_t operator[](int dimension) const noexcept;

  /// Return the local id of the provided logical item with respect
  /// to this group
  id_type
  get_logical_local_id(const sycl::s_item<dimensions> &idx) const noexcept;

  /// Returns the local linear id of the provided logical item with respect
  /// to this group
  linear_id_type
  get_logical_local_linear_id(const sycl::s_item<dimensions> &idx) const noexcept;

  /// Return the local id of the provided logical item in the 
  /// specified dimension with respect to this group
  size_t get_logical_local_id(const sycl::s_item<dimensions> &idx,
                              int dimension) const noexcept;

  /// Equivalent to get_logical_local_id()
  id_type get_local_id(const sycl::s_item<dimensions> &idx) const noexcept;

  /// Equivalent to get_logical_local_linear_id()
  linear_id_type
  get_local_linear_id(const sycl::s_item<dimensions> &idx) const noexcept;

  /// Equivalent to get_logical_local_id()
  size_t get_local_id(const sycl::s_item<dimensions> &idx,
                      int dimension) const noexcept;

  /// Equivalent to get_physical_local_id(); use
  /// get_physical_local_id() instead.
  [[deprecated("Use get_physical_local_id()")]]
  id_type get_local_id() const noexcept;

  /// Equivalent to get_physical_local_id(); use
  /// get_physical_local_id() instead.
  [[deprecated("Use get_physical_local_id()")]]
  id_type get_local_id(int dimension) const noexcept;

  /// Equivalent to get_physical_local_linear_id(); use
  /// get_physical_local_linear_id() instead.
  [[deprecated("Use get_physical_local_linear_id()")]]
  linear_id_type get_local_linear_id() const noexcept;

  /// Returns the id of the calling physical local item
  /// with respect to this group
  id_type get_physical_local_id() const noexcept;

  /// Returns the id of the calling physical local item
  /// in the specified dimension with respect to this group
  size_t get_physical_local_id(int dimension) const noexcept;

  /// Returns the linear id of the calling physical local item
  /// with respect to this group
  size_t get_physical_local_linear_id() const noexcept;

  /// Equivalent to get_logical_local_range(); use
  /// get_logical_local_range() instead.
  [[deprecated("Use get_logical_local_range()")]]
  range_type get_local_range() const noexcept;

  /// Equivalent to get_logical_local_range(); use
  /// get_logical_local_range() instead.
  [[deprecated("Use get_logical_local_range()")]]
  size_t get_local_range(int dimension) const noexcept;

  /// Equivalent to get_logical_local_linear_range(); use
  /// get_logical_local_linear_range() instead.
  [[deprecated("Use get_logical_local_linear_range()")]]
  size_t get_local_linear_range() const noexcept;

  /// Returns the size of the logical iteration space of this group
  range_type get_logical_local_range() const noexcept;

  /// Returns the size of the logical iteration space of this group
  /// in the specified dimension
  size_t get_logical_local_range(int dimension) const noexcept;

  /// Returns the size of the linear logical iteration space of this group
  size_t get_logical_local_linear_range() const noexcept;

  /// Returns the size of the physical iteration space of this group
  range_type get_physical_local_range() const noexcept;

  /// Returns the size of the physical iteration space of this group
  /// in the specified dimension
  size_t get_physical_local_range(int dimension) const noexcept;

  /// Returns the linear size of the physical iteration space of this group
  size_t get_physical_local_linear_range() const noexcept;

  /// Returns whether the calling item is the leader in the physical 
  /// iteration space. Should not be called within distribute_items().
  bool leader() const noexcept;

  /// Only available if the group is a work group; has same
  /// effect as the corresponding member function from sycl::group.
  /// See the SYCL specification for details.
  template <typename dataT>
  device_event
  async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                        size_t numElements) const noexcept;

  
  /// Only available if the group is a work group; has same
  /// effect as the corresponding member function from sycl::group.
  /// See the SYCL specification for details.
  template <typename dataT>
  device_event
  async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                        size_t numElements) const noexcept;

  /// Only available if the group is a work group; has same
  /// effect as the corresponding member function from sycl::group.
  /// See the SYCL specification for details.
  template <typename dataT>
  device_event
  async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                        size_t numElements, size_t srcStride) const noexcept;

  /// Only available if the group is a work group; has same
  /// effect as the corresponding member function from sycl::group.
  /// See the SYCL specification for details.
  template <typename dataT>
  device_event
  async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                        size_t numElements, size_t destStride) const noexcept;

  /// Only available if the group is a work group; has same
  /// effect as the corresponding member function from sycl::group.
  /// See the SYCL specification for details.
  template <typename... eventTN>
  void wait_for(eventTN...) const noexcept;

};
```

## Supported group algorithms

The following group algorithms are supported for scoped parallelism groups:
* `group_barrier()`
* `joint_*` functions (not yet implemented)
* `*_over_group` functions for arguments of private memory wrapper type obtained from `sycl::memory_environment()` (not yet implemented)

Group algorithms must be invoked outside of a `distribute_items()` call.

## class `s_private_memory`

This class was part of earlier versions of scoped parallelism. It is now deprecated and will be removed in the future. Use `sycl::memory_environment()` instead.

```c++
namespace sycl {
/// Scoped parallelism variant of sycl::private_memory.
/// Allows for sharing allocations in private memory of logical work
/// items between multiple distribute_items() invocations.
///
/// Must not be constructed inside distribute_items. Not user-constructible,
/// objects can be obtained using sycl::memory_environment().
///
/// Only available if SpGroup is a scoped parallelism group.
template<typename T, class SpGroup>
class s_private_memory
{
public:
  /// Construct object
  [[deprecated("Use sycl::memory_environment() instead")]]
  explicit s_private_memory(const SpGroup& grp);

  /// s_private_memory is not copyable
  s_private_memory(const s_private_memory&) = delete;
  s_private_memory& operator=(const s_private_memory&) = delete;

  /// Access allocation for specified logical work item.
  [[deprecated("Use sycl::memory_environment() instead")]]
  T& operator()(const s_item<SpGroup::dimensions>& idx) noexcept;
};

}

```

## class `local_memory<T, Group>` (deprecated)

This class was part of earlier versions of scoped parallelism. It is now deprecated and will be removed in the future. Use `sycl::memory_environment()` instead.

```c++
namespace sycl {

// Allocates a variable of type T in local memory.
template<class T, class Group>
class local_memory
{
public:
  using scalar_type = typename std::remove_extent<T>::type;

  // Only available if T is array. Returns reference to a single 
  // element of the array.
  [[deprecated("Use sycl::memory_environment() instead")]]
  scalar_type& operator[](std::size_t index) noexcept;

  // Return managed data in local memory
  [[deprecated("Use sycl::memory_environment() instead")]]
  T& operator()() noexcept;
};

}
```

## New class `s_item<int Dim>`

```c++
namespace sycl {

template<int Dim>
class s_item
{
public:
  /// Returns the global range of logical work items
  sycl::range<Dim> get_global_range() const;

  /// Returns the global range of logical work items
  size_t get_global_range(int dimension) const;

  /// Returns the global linear range of logical work items
  size_t get_global_linear_range() const;

  /// Returns the global id of the logical work item
  sycl::id<Dim> get_global_id() const;

  /// Returns the global id of the logical work item
  size_t get_global_id(int dimension) const;

  /// Returns the global linear id of the logical work item
  size_t get_global_linear_id() const;

  /// Returns the local range of the group nested innermost
  /// around the distribute_items() call.
  sycl::range<Dim> get_innermost_local_range() const noexcept;

  /// Returns the local range in the specified dimension of 
  /// the group nested innermost around the distribute_items() call.
  size_t get_innermost_local_range(int dimension) const noexcept;

  /// Returns the linear local range the group nested innermost 
  /// around the distribute_items() call.
  size_t get_innermost_local_linear_range() const noexcept;

  /// Returns the local id of this logical work item with respect
  /// to the group nested innermost around the distribute_items() call
  ///
  /// The get_innermost_local_id() family of functions may be more efficient
  /// than calling get_local_id(const Group&) family with the appropriate innermost
  /// group.
  sycl::id<Dim> get_innermost_local_id() const noexcept;

  /// Returns the local id in the specified dimension of this 
  /// logical work item with respect to the group nested innermost
  /// around the distribute_items() call
  ///
  /// The get_innermost_local_id() family of functions may be more efficient
  /// than calling get_local_id(const Group&) family with the appropriate innermost
  /// group.
  size_t get_innermost_local_id(int dimensions) const noexcept;

  /// Returns the local linear id in the of this 
  /// logical work item with respect to the group nested innermost
  /// around the distribute_items() call
  ///
  /// The get_innermost_local_id() family of functions may be more efficient
  /// than calling get_local_id(const Group&) family with the appropriate innermost
  /// group.
  size_t get_innermost_local_linear_id() const noexcept;

  /// Returns the local id of this logical work item with respect
  /// to the provided group.
  sycl::id<Dim> get_local_id(const Group& g) const noexcept;

  /// Returns the local id of this logical work item with respect
  /// to the provided group in the specified dimension.
  size_t get_local_id(const Group& g, int dimension) const noexcept;

  /// Returns the local linear id of this logical work item with respect
  /// to the provided group.
  size_t get_local_linear_id(const Group& g) const noexcept;

  /// Equivalent to grp.get_logical_local_range()
  template<class Group>
  sycl::range<Dim> get_local_range(const Group& grp) const noexcept;

  /// Equivalent to grp.get_local_range(dimension)
  template<class Group>
  size_t get_local_range(const Group& grp, int dimension) const noexcept;

  /// Equivalent to grp.get_logical_local_linear_range()
  template<class Group>
  size_t get_local_linear_range(const Group& grp) const noexcept;
};

}
```

