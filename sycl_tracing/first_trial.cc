#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/tracer_utils.hpp"
#include <iostream>
#include <sycl/sycl.hpp>

void cool_tracer(Tracer_utils::tracer_type type,
                 Tracer_utils::start_end state) {
  std::cout << "Hello World!" << std::endl;
}

int main() {

  sycl::gpu_selector selector;
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  auto dev = q.get_device();

  std::cout << "Running on device: " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  // Tracer_utils::initialize_tracer(cool_tracer);

  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      int i = 0;
      for (int j = 0; j < 100; j++) {
        i++;
      }
    });
  });

  std::cout << "Hello World!" << std::endl;
}
