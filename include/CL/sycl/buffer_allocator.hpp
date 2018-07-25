#ifndef SYCU_BUFFER_ALLOCATOR_HPP
#define SYCU_BUFFER_ALLOCATOR_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <limits>
#include <utility>

#include "backend/backend.hpp"
#include "exception.hpp"

namespace cl {
namespace sycl {


template <typename T>
using buffer_allocator = std::allocator<T>;

// ToDo Image allocator

#ifdef __NVCC__

template<class T>
class svm_allocator
{
public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  template <class U> struct rebind {
    using other = svm_allocator<U>;
  };

  using propagate_on_container_move_assignment = std::true_type;

  svm_allocator() noexcept {}

  svm_allocator (const allocator& alloc) noexcept {}

  template <class U>
  svm_allocator (const allocator<U>& alloc) noexcept {}

  pointer address ( reference x ) const noexcept
  { return &x; }

  const_pointer address ( const_reference x ) const noexcept
  { return &x; }

  pointer allocate(size_type n, svm_allocator<void>::const_pointer hint=0)
  {
    void* ptr = nullptr;
    cudaError_t result = cudaMallocManaged(&ptr, n * sizeof(T));

    if(result != cudaSuccess)
      throw memory_allocation_error{"SVM allocator: bad allocation", static_cast<hipError_t>{result}};

    return reinterpret_cast<pointer>(ptr);
  }

  void deallocate (pointer p, size_type n)
  {
    cudaError_t result = cudaFree(p);
    if(result != cudaSuccess)
      throw runtime_error{"SVM allocator: Could not free memory", static_cast<hipError_t>{result}};
  }

  size_type max_size() const noexcept
  {
    return std::numeric_limits<size_type>::max();
  }

  template <class U, class... Args>
  void construct (U* p, Args&&... args)
  {
    *p = U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy (U* p)
  { p->~U(); }
};

#endif

}
}


#endif 
