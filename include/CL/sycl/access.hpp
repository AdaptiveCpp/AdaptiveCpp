#ifndef SYCU_ACCESS_HPP
#define SYCU_ACCESS_HPP

namespace cl {
namespace sycl {
namespace access {

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class placeholder {
  false_t,
  true_t
};

}


}
}


#endif
