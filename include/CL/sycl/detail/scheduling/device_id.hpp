/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_DEVICE_ID_HPP
#define HIPSYCL_DEVICE_ID_HPP

#include <functional>

#include "../../backend/backend.hpp"

namespace cl {
namespace sycl {
namespace detail {

class device_id
{
public:
  device_id() = default;
  device_id(const device_id&) = default;
  device_id(backend b, int id);
  
  bool is_host() const;
  backend get_backend() const;
  int get_id() const;

  friend bool operator==(const device_id& a, const device_id& b)
  {
    return a._backend == b._backend && 
           a._device_id == b._device_id;
  }

  friend bool operator!=(const device_id& a, const device_id& b)
  {
    return !(a == b);
  }
private:
  backend _backend;
  int _device_id;
};

}
}
}


namespace std {

template <>
struct hash<cl::sycl::detail::device_id>
{
  std::size_t operator()(const cl::sycl::detail::device_id& k) const
  {
    return hash<int>()(static_cast<int>(k.get_backend())) ^ 
          (hash<int>()(k.get_id()) << 1);
  }
};

}

#endif
