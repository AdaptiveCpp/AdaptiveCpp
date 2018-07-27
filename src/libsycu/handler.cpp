#include "CL/sycl/handler.hpp"
#include "CL/sycl/queue.hpp"

namespace cl {
namespace sycl {

handler::handler(const queue& q)
: _queue{new queue{q}}
{}

}
}
