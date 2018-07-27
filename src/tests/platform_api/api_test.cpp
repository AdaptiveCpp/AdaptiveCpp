#include <iostream>
#include <CL/sycl.hpp>

int main()
{
  cl::sycl::platform platform;
  auto devs = platform.get_devices();

  for(const auto& d : devs)
    std::cout << "Found device " << d.get_device_id() << std::endl;

  cl::sycl::queue q;
  std::cout << "Created queue on GPU: " << ((q.is_host() == false) ? "true" : "false")
                                        << std::endl;
}
