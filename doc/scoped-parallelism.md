# Scoped parallelism

## Introduction

Scoped parallelism provides a novel way to formulate kernels. Scoped parallelism aims at combining the advantages of hierarchical parallelism with those of classical `nd_range` parallel for kernels:
* It is performance portable and efficiently implementable on the host device (contrary to `nd_range` parallel for);
* It has a clear and well-defined behavior and is easy to implement and maintain (contrary to hierarchical parallelism);
* It does not present the user with performance surprises (contrary to both hierarchical parallelism and `nd_range` parallel for on the CPU)

Scoped parallelism acknowledges that an implementation may achieve better performance by employing a different degree of parallelization within a work group compared to the user-requested work group size.
Like hierarchical parallelism, scoped parallelism kernels consist of two scopes. But, unlike hierarchical parallelism, within the outer scope the implementation-defined physical parallelism is exposed. In the inner scope, the user-provided logical parallelism is exposed.

Structurally, within a work group scoped parallelism is similar to patterns found in other programming models like OpenMP, where a (by default) implementation-defined number of threads is first spawned with `#pragma omp parallel` and later on, a user-provided parallel iteration space is distributed across the physical one using `#pragma omp for`.
Scoped parallelism works analogously: `handler::parallel()` launches a parallel kernel, and `group::distribute_for()` distributes the user-provided iteration space across the execution resources.

Scoped parallelism exposes both work items to query the position in the physical and logical iteration spaces: The `physical_item` and the `logical_item`. Additionally, `sub_group` is exposed as well, providing information about the warps/wavefronts on GPUs.

Because scoped parallelism enforces explicit information about where to allocate variables from the user, performance surprises as in hierarchical parallel for are prevented. In particular, simple variable declarations in the outer scope will not be allocated in local memory (as in hierarchical parallelism), but in the private memory of the physical work item which will always be efficient.

## Example code

```c++
#include <SYCL/sycl.hpp>
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
    [=](sycl::group<1> grp, sycl::physical_item<1> phys_idx){
      // Inside the parallel scope, the degree of parallelism is implementation-defined
      // the implementation can use whatever is most efficient for hardware/backend.
      // In hipSYCL CPU, this would be executed by a single thread on CPU
      // and Group_size threads on hipSYCL GPU
      // phys_idx contains information about the position in the (implementation-defined)
      // physical iteration space

      // Local memory in parallel scope must be allocated explicitly as such
      sycl::local_memory<int [Group_size]> scratch{grp};
      // Same goes for private_memory, if required (here only dummy example)
      // This will allocate memory in the private memory of the _logical_ work item.
      // See the existing SYCL documentation on private_memory for more information.
      sycl::private_memory<int> dummy{grp};
      
      // Variables not explicitly marked as either will be allocated in private
      // memory of the _physical_ work item (see the for loop below)

      // `distribute_for` distributes the logical, user-provided iteration space across the physical one. 
      grp.distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx){
          scratch[idx.get_local_id(0)] = data_accessor[idx.get_global_id(0)];
          // Can execute code for a single item of the subgroup:
          sg.single_item([&](){
            //...
          });
      }); // implicit barrier at the end by default

      // Variables inside the parallel scope that are not explicitly local or private memory
      // are allowed, if they are not modified from inside `distribute_for()` scope.
      // The SYCL implementation will allocate those in private memory of the physical thread,
      // so they will always be efficient. This implies that the user should not attempt to assign values
      // per logical work item, since they are allocated per physical item.
      for(int i = Group_size / 2; i > 0; i /= 2){
        grp.distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx){
          size_t lid = idx.get_local_id(0);
          if(lid < i)
            scratch[lid] += scratch[lid+i];
        });
      }
      
      grp.single_item([&](){
        data_accessor[grp.get_id(0)*Group_size] = scratch[0];
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
// Spawns a new parallel region. f must have the signature 
// void(group<dimensions>, physical_item<dimensions>)
template <typename KernelName, typename KernelFunctionType, int dimensions>
void handler::parallel(range<dimensions> numWorkGroups,
                      range<dimensions> workGroupSize,
                      KernelFunctionType f);
```

### New functions in `sycl::group`

```c++
// Distribute the execution of f across the available parallel resources.
// f must have the signature void(sub_group, logical_item<dimensions>)
template<typename distributeForFunctionT>
void group::distribute_for(distributeForFunctionT f) const;

// Execute f for a single item of the group. f has the signature void().
template<class Function>
void group::single_item(Function f);
```

## New functions in `sub_group`

```c++
// Execute f for a single item of the sub_group. f has the signature void().
template<class F>
void sub_group::single_item(F f);
```

## New functions `private_memory`

```c++
// Return managed data for the logical work item
T& operator()(const logical_item<Dim> &item) noexcept;
};
```

## class `local_memory<T>`

```c++
// Allocates a variable of type T in local memory.
template<class T>
class local_memory
{
public:
  using scalar_type = typename std::remove_extent<T>::type;

  // Constructor
  template<int Dim>
  local_memory(group<Dim>&);

  // Only available if T is array. Returns reference to a single 
  // element of the array.
  scalar_type& operator[](std::size_t index) noexcept;

  // Return managed data in local memory
  T& operator()() noexcept;
};
```

## class `physical_item<int Dim>`

```c++
template<int Dim>
class physical_item
{
public:
  // Returns the global range of physical work items
  sycl::range<Dim> get_global_range() const;

  // Returns the global range of physical work items
  size_t get_global_range(int dimension) const;

  // Returns the global id of the physical work item
  sycl::id<Dim> get_global_id() const;

  // Returns the global id of the physical work item
  size_t get_global_id(int dimension) const;

  // Returns the global linear id of the physical work item
  size_t get_global_linear_id() const;

  // Returns the local range of physical work items
  sycl::range<Dim> get_local_range() const;

  // Returns the local range of physical work items
  size_t get_local_range(int dimension) const;

  // Returns the local id of the physical work item
  sycl::id<Dim> get_local_id() const;

  // Returns the local id of the physical work item
  size_t get_local_id(int dimension) const;
  
  // Returns the local linear id of the physical work item
  size_t get_local_linear_id() const;
};

```

## class `logical_item<int Dim>`

```c++
template<int Dim>
class logical_item
{
public:
  // Returns the global range of logical work items
  sycl::range<Dim> get_global_range() const;

  // Returns the global range of logical work items
  size_t get_global_range(int dimension) const;

  // Returns the global id of the logical work item
  sycl::id<Dim> get_global_id() const;

  // Returns the global id of the logical work item
  size_t get_global_id(int dimension) const;

  // Returns the global linear id of the logical work item
  size_t get_global_linear_id() const;

  // Returns the local range of logical work items
  sycl::range<Dim> get_local_range() const;

  // Returns the local range of logical work items
  size_t get_local_range(int dimension) const;

  // Returns the local id of the logical work item
  sycl::id<Dim> get_local_id() const;

  // Returns the local id of the logical work item
  size_t get_local_id(int dimension) const;
  
  // Returns the local linear id of the logical work item
  size_t get_local_linear_id() const;
};

```
