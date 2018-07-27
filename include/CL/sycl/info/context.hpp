#ifndef SYCU_INFO_CONTEXT_HPP
#define SYCU_INFO_CONTEXT_HPP

#include "param_traits.hpp"

namespace cl {
namespace sycl {
namespace info {

enum class context : int {
  reference_count,
  platform,
  devices
};



}
}
}



#endif
