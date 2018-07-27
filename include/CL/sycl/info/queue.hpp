#ifndef SYCU_INFO_QUEUE_HPP
#define SYCU_INFO_QUEUE_HPP

namespace cl {
namespace sycl {
namespace info {

enum class queue : int
{
  context,
  device,
  reference_count
};

}
}
}

#endif
