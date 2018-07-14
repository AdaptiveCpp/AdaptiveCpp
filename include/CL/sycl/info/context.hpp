#ifndef SYCU_INFO_CONTEXT_HPP
#define SYCU_INFO_CONTEXT_HPP

#include "param_traits.hpp"

namespace cl {
namespace sycl {

/** \addtogroup execution Platforms, contexts, devices and queues
    @{
*/
namespace info {

/** Context information descriptors

    \todo Should be unsigned int to be consistent with others?
*/
enum class context : int {
  reference_count,
  platform,
  devices
};



}
}
}



#endif
