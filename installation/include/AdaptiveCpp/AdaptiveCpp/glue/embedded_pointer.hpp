/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
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
namespace detail {

template<class T>
inline T random_number() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen{rd()};
  thread_local std::uniform_int_distribution<T> distribution{0};

  return distribution(gen);
}

}

struct unique_id {
  static constexpr std::size_t num_components = 2;

  ACPP_UNIVERSAL_TARGET
  unique_id(uint64_t init_value) {
    for(std::size_t i=0; i < num_components; ++i){
      id[i] = init_value;
    }
  }

  ACPP_UNIVERSAL_TARGET
  unique_id() {
    __acpp_if_target_host(
      uint64_t ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();

      uint64_t random_number = detail::random_number<uint64_t>();

      char* ns_bytes = reinterpret_cast<char*>(&ns);
      char* rnd_bytes = reinterpret_cast<char*>(&random_number);

      for(int i = 0; i < sizeof(uint64_t); ++i) {
        char *id_bytes = reinterpret_cast<char*>(&id[0]);
        id_bytes[2 * i    ] = ns_bytes[i];
        id_bytes[2 * i + 1] = rnd_bytes[i];
      }
    );
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator==(const unique_id& a, const unique_id& b) {
    for(std::size_t i = 0; i < num_components; ++i) {
      if(a.id[i] != b.id[i])
        return false;
    }
    return true;
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator!=(const unique_id& a, const unique_id& b) {
    return !(a == b);
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return id[0] ^ id[1];
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
  static_assert(sizeof(unique_id) == 2 * sizeof(void*));

  embedded_pointer() {
    __acpp_if_target_host(
      unique_id uid;
      std::memcpy(&_ptrs[0], &uid, sizeof(unique_id));
    );
  }

  embedded_pointer(const embedded_pointer&) = default;

  ACPP_UNIVERSAL_TARGET
  T* get() const {

    return reinterpret_cast<T*>(_ptrs[0]);
  }

  ACPP_UNIVERSAL_TARGET
  unique_id get_uid() const {
    // Initialize to 0 to avoid generating new id
    unique_id id{0};
    std::memcpy(&id, &_ptrs[0], sizeof(unique_id));
    return id;
  }

  // this is only necessary when no initialization
  // from within a kernel blob happens
  void explicit_init(void* ptr) {
    _ptrs[0] = ptr;
    _ptrs[1] = 0;
  }
  
  ACPP_UNIVERSAL_TARGET
  friend bool operator==(const embedded_pointer &a, const embedded_pointer &b) {
    return a._ptrs[0] == b._ptrs[0] && a._ptrs[1] == b._ptrs[1];
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator!=(const embedded_pointer &a, const embedded_pointer &b) {
    return !(a == b);
  }

private:
  void* _ptrs [2];
};

struct kernel_blob {

  static bool initialize_embedded_pointer(void *b, std::size_t blob_size,
                                          const unique_id &pointer_id,
                                          const void *ptr) {

    char* blob_ptr = reinterpret_cast<char*>(b);
    bool found = false;

    // TODO: We could try to optimize this by only
    // looking at offsets that are properly aligned
    // if we are sure there are no cases where this
    // might go wrong
    for(int i = 0; i + sizeof(unique_id) <= blob_size;) {
    
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
