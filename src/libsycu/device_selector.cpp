#include "CL/sycl/device_selector.hpp"

namespace cl {
namespace sycl {


device device_selector::select_device() const
{
  auto devices = device::get_devices();
  if(devices.size() == 0)
    throw platform_error{"No available devices!"};

  int best_score = std::numeric_limits<int>::min();
  device candidate;
  for(const device& d : devices)
  {
    int current_score = (*this)(d);
    if(current_score > best_score)
    {
      best_score = current_score;
      candidate = d;
    }
  }
  return candidate;
}

}
}
