#ifndef SYCU_PARAM_TRAITS_HPP
#define SYCU_PARAM_TRAITS_HPP

#include "../types.hpp"

namespace cl {
namespace sycl {
namespace info {

template <typename T, T Param>
struct param_traits {};

#define SYCU_PARAM_TRAIT_RETURN_VALUE(T, param, ret_value) \
  template<> \
  struct param_traits<T, param> \
  { using return_type = ret_value; };

}
}
}

#endif
