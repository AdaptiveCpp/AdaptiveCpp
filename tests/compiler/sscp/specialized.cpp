// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s


#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

class test_kernel {
public:
  template <class T1, class T2, class T3, class T4, class T5, class T6,
            class T7>
  test_kernel(uint64_t *data, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7)
      : _data{data}, _v1{static_cast<uint8_t>(v1)},
        _v2{static_cast<uint16_t>(v2)}, _v3{static_cast<uint32_t>(v3)},
        _v4{static_cast<uint64_t>(v4)}, _v5{static_cast<float>(v5)},
        _v6{static_cast<double>(v6)}, _v7{reinterpret_cast<int*>(v7)} {}

  void operator()() const noexcept {
    *_data *=  static_cast<uint64_t>(_v1); 
    *_data *=  static_cast<uint64_t>(_v2);
    *_data *=  static_cast<uint64_t>(_v3);
    *_data *=  static_cast<uint64_t>(_v4);
    *_data *=  static_cast<uint64_t>(_v5);
    *_data *=  static_cast<uint64_t>(_v6);
    *_data +=  reinterpret_cast<uint64_t>((int*)_v7);
  }

private:
  uint64_t* _data;
  sycl::specialized<uint8_t> _v1;
  sycl::specialized<uint16_t> _v2;
  sycl::specialized<uint32_t> _v3;
  sycl::specialized<uint64_t> _v4;
  sycl::specialized<float> _v5;
  sycl::specialized<double> _v6;
  sycl::specialized<int*> _v7;
};

int main () {
  sycl::queue q = get_queue();

  uint64_t* data = sycl::malloc_shared<uint64_t>(1, q);

  *data = 1;
  q.single_task(test_kernel{data, 0, 2, 3, 4, 5, 6, 7}).wait();
  // CHECK: 7
  std::cout << *data << std::endl;

  *data = 1;
  q.single_task(test_kernel{data, 1, 2, 3, 1, 1, 1, 7}).wait();
  // CHECK: 13
  std::cout << *data << std::endl;


  sycl::free(data, q);
  
}
