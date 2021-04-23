/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_GLUE_MODULE_HPP
#define HIPSYCL_GLUE_MODULE_HPP


#include <vector>
#include <array>
#include <string>
#include "hipSYCL/runtime/device_id.hpp"

struct __hipsycl_cuda_embedded_object {
  std::string target;
  std::string data;
};

#ifdef __HIPSYCL_MULTIPASS_CUDA_HEADER__
// The header included here should define an object __hipsycl_cuda_bundle as follows:
// static const unsigned long long __hipsycl_cuda_bundle_id = ...;
// static std::array __hipsycl_cuda_bundle {
//   __hipsycl_cuda_embedded_object {std::string{"target1"}, std::string{"ptx
//   code1"}},
//   __hipsycl_cuda_embedded_object {std::string{"target2"}, std::string{"ptx
//   code2"}},
//   ...
// };

 #include __HIPSYCL_MULTIPASS_CUDA_HEADER__
#else
// If we don't have CUDA embedded objects, construct empy bundle
static const unsigned long long __hipsycl_cuda_bundle_id = 0;
static std::array<__hipsycl_cuda_embedded_object, 0> __hipsycl_cuda_bundle;
#endif

#ifdef __HIPSYCL_MULTIPASS_SPIRV_HEADER__
 #include __HIPSYCL_MULTIPASS_SPIRV_HEADER__
#else
static const unsigned long long __hipsycl_spirv_bundle_id = 0;
static std::array<std::string, 0> __hipsycl_spirv_bundle;
#endif

namespace hipsycl {
namespace glue {

namespace this_module {

// These functions cannot be moved to a struct because member functions
// cannot have static linkage.
template <rt::backend_id Backend, class Handler>
static void for_each_object(Handler &&h) {
  if constexpr(Backend == rt::backend_id::cuda) {
    for (const auto &obj : __hipsycl_cuda_bundle) {
      h(obj);
    }
  } else if constexpr(Backend == rt::backend_id::level_zero) {
    for (const auto &obj : __hipsycl_spirv_bundle) {
      h(obj);
    }
  }
}

template <rt::backend_id Backend, class Handler>
static void for_each_target(Handler &&h) {
  if constexpr(Backend == rt::backend_id::cuda) {
    for (const auto &obj : __hipsycl_cuda_bundle) {
      h(obj.target);
    }
  } else if constexpr(Backend == rt::backend_id::level_zero) {
    h(std::string{"spirv"});
  }
}

template <rt::backend_id Backend>
static const std::string *get_code_object(const std::string &target) {
  if constexpr (Backend == rt::backend_id::cuda) {
    for (const auto &obj : __hipsycl_cuda_bundle) {
      if (obj.target == target)
        return &(obj.data);
    }
  } else if constexpr (Backend == rt::backend_id::level_zero) {
    if(__hipsycl_spirv_bundle.size() > 0) {
      return &(__hipsycl_spirv_bundle[0]);
    }
  }
  return nullptr;
}

template <rt::backend_id Backend>
static constexpr std::size_t get_num_objects() {
  if constexpr (Backend == rt::backend_id::cuda) {
    return __hipsycl_cuda_bundle.size();
  } else if constexpr(Backend == rt::backend_id::level_zero) {
    return __hipsycl_spirv_bundle.size();
  }else {
    return 0;
  }
}

template <rt::backend_id Backend>
static constexpr std::size_t get_module_id() {
  if constexpr (Backend == rt::backend_id::cuda) {
    return __hipsycl_cuda_bundle_id;
  } else if constexpr(Backend == rt::backend_id::level_zero) {
    return __hipsycl_spirv_bundle_id;
  }else {
    return 0;
  }
}

} // this_module

}
} // namespace hipsycl

#endif
