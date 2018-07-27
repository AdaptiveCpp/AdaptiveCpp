#ifndef SYCU_ACCESSOR_HPP
#define SYCU_ACCESSOR_HPP

#include "access.hpp"

namespace cl {
namespace sycl {

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget = access::target::global_buffer,
         access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor
{

};


}
}

#endif
