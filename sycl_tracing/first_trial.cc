#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include <iostream>
#include <sycl/sycl.hpp>

// void cool_tracer(Tracer_utils::tracer_type type,
//                  Tracer_utils::start_end state) {
//   std::cout << "Hello World!" << std::endl;
// }

int main() {

  sycl::gpu_selector selector;

  sycl::queue q{selector};
  //, sycl::property_list{sycl::property::queue::in_order{}}};

  auto ctx = q.get_context();

  sycl::host_selector selector2;
  sycl::queue q2{ctx, selector2, sycl::property_list{sycl::property::queue::in_order{}}};

  auto dev = q.get_device();
  auto dev2 = q2.get_device();

  std::cout << "Running on device: " << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Running on device with q2: " << dev2.get_info<sycl::info::device::name>()
            << std::endl;

  auto context1 = q.get_context();
  auto context2 = q2.get_context();

  if (context2 == context1) {
    std::cout << "The contexts are the same:" << std::endl;
  } else {
    std::cout << "The contexts are not the same: " << std::endl;
  }

  // Tracer_utils::initialize_tracer(cool_tracer);

  std::array<int, 100> numbers;
  for (int i = 1; i <= 100; i++)
    numbers[i - 1] = i;

  int *numbers_shared = sycl::malloc_shared<int>(100, q);

  auto e0 = q2.single_task([=] {
    for (int i = 1; i <= 100; i++)
      numbers_shared[i - 1] = i;
  });

  // q2.wait();

  auto e1 = q.memset(numbers_shared, 0, sizeof(int) * 100, e0);
  // q.wait();
  auto e2 = q2.fill(numbers_shared, 42, 100, e1);
  q2.wait();
  //  // e.wait();
  //
  // q.copy(numbers_device, numbers.data(), 100);
  //
  //  // q.wait();
  //
  auto e3 = q.submit([&](sycl::handler &h) {
    h.depends_on(e2);
    h.single_task([=]() {
      int i = 0;
      for (int j = 0; j < 100; j++) {
        i++;
      }
    });
  });

  // q.wait();

  q.parallel_for(sycl::range<1>(10), e3, [=](sycl::id<1> I) { const int i = 0; }).wait();
  // q.wait();

  std::cout << "Hello World!" << std::endl;
}
