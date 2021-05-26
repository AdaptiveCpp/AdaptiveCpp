/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_EMBEDDED_POINTER_HPP
#define HIPSYCL_EMBEDDED_POINTER_HPP

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"

#include <cstring>
#include <iterator>
#include <memory>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>

namespace hipsycl {
namespace glue {


struct unique_id {
  static constexpr std::size_t num_components = 2;

  HIPSYCL_UNIVERSAL_TARGET
  unique_id(uint64_t init_value) {
    for(std::size_t i=0; i < num_components; ++i){
      id[i] = init_value;
    }
  }

  HIPSYCL_UNIVERSAL_TARGET
  unique_id() {
#ifndef SYCL_DEVICE_ONLY
    uint64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    static std::random_device rd;
    static std::mt19937 gen{rd()};
    static std::uniform_int_distribution<uint64_t> distribution{0};

    uint64_t random_number = distribution(gen);

    char* ns_bytes = reinterpret_cast<char*>(&ns);
    char* rnd_bytes = reinterpret_cast<char*>(&random_number);

    for(int i = 0; i < sizeof(uint64_t); ++i) {
      char *id_bytes = reinterpret_cast<char*>(&id[0]);
      id_bytes[2 * i    ] = ns_bytes[i];
      id_bytes[2 * i + 1] = rnd_bytes[i];
    }
#endif
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const unique_id& a, const unique_id& b) {
    for(std::size_t i = 0; i < num_components; ++i) {
      if(a.id[i] != b.id[i])
        return false;
    }
    return true;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const unique_id& a, const unique_id& b) {
    return !(a == b);
  }

  unique_id(const unique_id&) = default;
  unique_id& operator=(const unique_id&) = default;

  alignas(8) uint64_t id [num_components];
};

inline std::ostream &operator<<(std::ostream &ostr, const unique_id &id) {
  ostr << id.id[0] << "-" << id.id[1];
  return ostr;
}

template<class T>
class embedded_pointer {
public:
  HIPSYCL_UNIVERSAL_TARGET
  T* get() const {
    static_assert(sizeof(T*) == sizeof(uint64_t));
    static_assert(sizeof(T*) == sizeof(unique_id) / 2);

    return reinterpret_cast<T*>(_uid.id[0]);
  }

  HIPSYCL_UNIVERSAL_TARGET
  const unique_id& get_uid() const {
    return _uid;
  }

  // this is only necessary when no initialization
  // from within a kernel blob happens
  void explicit_init(void* ptr) {
    _uid.id[0] = reinterpret_cast<uint64_t>(ptr);
    _uid.id[1] = 0;
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const embedded_pointer &a, const embedded_pointer &b) {
    return a._uid == b._uid;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const embedded_pointer &a, const embedded_pointer &b) {
    return !(a == b);
  }

private:
  unique_id _uid;
};

struct kernel_blob {
  template <class Blob>
  static bool initialize_embedded_pointer(Blob &b, const unique_id &pointer_id,
                                          void *ptr) {

    char* blob_ptr = reinterpret_cast<char*>(&b);
    bool found = false;

    // TODO: We could try to optimize this by only
    // looking at offsets that are properly aligned
    // if we are sure there are no cases where this
    // might go wrong
    for(int i = 0; i + sizeof(unique_id) <= sizeof(Blob);) {
    
      char* chunk_ptr = blob_ptr + i;

      if(std::memcmp(chunk_ptr, &pointer_id, sizeof(unique_id))==0) {
        HIPSYCL_DEBUG_INFO << "Identified embedded pointer with uid "
                          << pointer_id << " in kernel blob, setting to "
                          << ptr << std::endl;
        // Zero out detected embedded pointer
        std::memset(chunk_ptr, 0, sizeof(unique_id));
        // Set first 8 bytes of embedded pointer to ptr
        std::memcpy(chunk_ptr, &ptr, sizeof(void*));
        
        found = true;
        // we cannot stop after having found an embedded pointer
        // because there might be multiple copies of the same pointer
        // and we need to initialize them all.
        // This can happen if multiple copies of the same accessor
        // are captured.
        i += sizeof(unique_id);
      } else {
        ++i;
      }
    
    }

    return found;
  }
};

} // glue
} // hipsycl


#endif
