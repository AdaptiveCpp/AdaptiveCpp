#ifndef SYCU_BACKEND_HPP
#define SYCU_BACKEND_HPP

#ifdef SYCU_USE_HIP
#include <hip/hip_runtime.h>
#else
#include "../contrib/include/hip/hip_runtime.h"
#endif

//#ifdef __HIP_DEVICE_COMPILE__
//#define __SYCU_DEVICE__
//#endif

#endif
