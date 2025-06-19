#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include <iostream>
#include <sycl/sycl.hpp>

// void cool_tracer(Tracer_utils::tracer_type type,
//                  Tracer_utils::start_end state) {
//   std::cout << "Hello World!" << std::endl;
// }

int main() {

  sycl::cpu_selector selector;
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  auto dev = q.get_device();

  std::cout << "Running on device: " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  // Tracer_utils::initialize_tracer(cool_tracer);

  std::array<int, 100> numbers;
  for (int i = 1; i <= 100; i++)
    numbers[i - 1] = i;

  int *numbers_device = sycl::malloc_device<int>(100, q);
  q.memcpy(numbers_device, numbers.data(), sizeof(int) * 100);
  q.memset(numbers_device, 0, sizeof(int) * 100);
  q.fill(numbers_device, 42, 100);
  q.copy(numbers_device, numbers.data(), 100);

  q.wait();

  q.submit([&](sycl::handler &h) {
     h.single_task([=]() {
       int i = 0;
       for (int j = 0; j < 100; j++) {
         i++;
       }
     });
   }).wait();

  q.parallel_for(sycl::range<1>(10), [=](sycl::id<1> I) {
     const int i = 0;
   }).wait();

  std::cout << "Hello World!" << std::endl;
}
