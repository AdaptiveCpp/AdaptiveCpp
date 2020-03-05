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

namespace hipsycl {
namespace rt {

enum class hardware_platform
{
  rocm,
  cuda,
  cpu
};

enum class api_platform
{
  hip,
  openmp_cpu
};

enum class backend_id
{
  hip,
  openmp_cpu
};

struct backend_descriptor
{
  backend_id id;
  api_platform sw_platform;
  hardware_platform hw_platform;

  friend bool operator==(const backend_descriptor &a,
                         const backend_descriptor &b) {
    return a.id == b.id;
  }
};

class device_id
{
public:
  device_id() = default;
  device_id(const device_id&) = default;
  device_id(backend_descriptor b, int id);
  
  bool is_host() const;
  backend_id get_backend() const;
  backend_descriptor get_full_backend_descriptor() const;
  
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
  backend_descriptor _backend;
  int _device_id;
};

}
}


namespace std {

template <>
struct hash<hipsycl::rt::device_id>
{
  std::size_t operator()(const hipsycl::rt::device_id& k) const
  {
    return hash<int>()(static_cast<int>(k.get_backend())) ^ 
          (hash<int>()(k.get_id()) << 1);
  }
};

}

#endif
