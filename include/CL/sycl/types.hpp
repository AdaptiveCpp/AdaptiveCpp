#ifndef SYCU_TYPES_HPP
#define SYCU_TYPES_HPP

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <memory>
#include <exception>
#include <mutex>

namespace cl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
template<class T, class Alloc = std::allocator<T>>
using vector_class = std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
using string_class = std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
template<class Func>
using function_class = std::function<Func>;
#endif

#ifndef CL_SYCL_NO_STD_MUTEX
using mutex_class = std::mutex;
#endif

#ifndef CL_SYCL_NO_STD_UNIQUE_PTR
template<class T>
using unique_ptr_class = std::unique_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_SHARED_PTR
template<class T>
using shared_ptr_class = std::shared_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_WEAK_PTR
template<class T>
using weak_ptr_class = std::weak_ptr<T>;
#endif

#ifndef CL_SYCL_NO_HASH
template<class T>
using hash_class = std::hash<T>;
#endif

using exception_ptr_class = std::exception_ptr;


using exception_ptr = exception_ptr_class;
using exception_list = vector_class<exception_ptr>;
using async_handler = function_class<void(cl::sycl::exception_list)>;

// \todo Better use uint32_t etc
using cl_uchar = unsigned char;
using cl_ushort = unsigned short;
using cl_uint = unsigned;
using cl_ulong = unsigned long long;

using cl_char = char;
using cl_short = short;
using cl_int = int;
using cl_long = long long;

using cl_float = float;
using cl_double = double;


}
}

// Only pull typedefs into global namespace if the OpenCL headers
// defining them haven't yet been pulled in
#ifndef CL_TARGET_OPENCL_VERSION
using cl::sycl::cl_uchar;
using cl::sycl::cl_ushort;
using cl::sycl::cl_uint;
using cl::sycl::cl_ulong;

using cl::sycl::cl_char;
using cl::sycl::cl_short;
using cl::sycl::cl_int;
using cl::sycl::cl_long;

using cl::sycl::cl_float;
using cl::sycl::cl_double;
#endif

#endif
