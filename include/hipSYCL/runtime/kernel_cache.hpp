/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay and contributors
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

#include <string>
#include <unordered_map>
#include <mutex>
#include <cassert>
#include <memory>
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/common/small_map.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"

#ifndef HIPSYCL_RT_KERNEL_CACHE_HPP
#define HIPSYCL_RT_KERNEL_CACHE_HPP

namespace hipsycl {
namespace rt {

enum class code_format {
  ptx,
  amdgpu,
  spirv,
  native_isa
};

enum class code_object_state {
  executable,
  compiled,
  device_ir,
  generic_ir,
  source
};

using hcf_object_id = std::size_t;

class code_object {
public:
  virtual ~code_object() {}
  virtual code_object_state state() const = 0;
  virtual code_format format() const = 0;
  virtual backend_id managing_backend() const = 0;
  virtual hcf_object_id hcf_source() const = 0;
  virtual std::string target_arch() const = 0;
  
  virtual std::vector<std::string> supported_backend_kernel_names() const = 0;
  virtual bool contains(const std::string& backend_kernel_name) const = 0;
};

class kernel_cache {
public:
  using code_object_ptr = std::unique_ptr<code_object>;
  using code_object_index_t = std::size_t;
  using kernel_name_index_t = std::size_t;

  static kernel_cache& get();
  
  void register_hcf_object(const common::hcf_container& obj);

  const kernel_name_index_t*
  get_global_kernel_index(const std::string &kernel_name) const;

  const common::hcf_container* get_hcf(hcf_object_id obj) const;

  template<class KernelT>
  void register_kernel() {
    std::string name = get_global_kernel_name<KernelT>();

    _kernel_names.push_back(name);
    kernel_name_index_t idx = _kernel_names.size() - 1;
    _kernel_index_map[name] = idx;
  }

  template<class KernelT>
  std::string get_global_kernel_name() const {
    return typeid(KernelT).name();
  }

  template<class KernelT>
  kernel_name_index_t get_global_kernel_index() const {
    return get_global_kernel_index(get_global_kernel_name<KernelT>());
  }

  template<class F>
  void for_each_hcf_object(F&& f) const {
    for(const auto& v : _hcf_objects) {
      f(v.first, v.second);
    }
  }

  // Retrieve object for provided kernel and backend, or nullptr
  // if not found.
  template<class Predicate>
  const code_object* get_code_object(kernel_name_index_t kernel_index,
    const std::string& backend_kernel_name,
    backend_id b, Predicate&& object_selector) {

    std::lock_guard<std::mutex> lock{_mutex};
    return get_code_object_impl(kernel_index, backend_kernel_name, b,
                                object_selector);
  }

  template<class Predicate>
  const code_object* get_code_object(kernel_name_index_t kernel_index,
    const std::string& backend_kernel_name,
    backend_id b, hcf_object_id source_object, Predicate&& object_selector) {

    std::lock_guard<std::mutex> lock{_mutex};

    auto pred = [&](const code_object *obj) {
      if (obj->hcf_source() != source_object)
        return false;

      return object_selector(obj);
    };

    return get_code_object_impl(kernel_index, backend_kernel_name, b,
                                pred);
  }

  template <class Constructor, class Predicate>
  const code_object *get_or_construct_code_object(
      kernel_name_index_t kernel_index, const std::string &backend_kernel_name,
      backend_id b, Predicate &&object_selector, Constructor &&c) {

    std::lock_guard<std::mutex> lock{_mutex};

    return get_or_construct_code_object_impl(kernel_index, backend_kernel_name,
                                             b, object_selector, c);
  }

  // Only to be used within a Constructor passed to get_or_construct_code_object!
  template <class Constructor, class Predicate>
  const code_object *recursive_get_or_construct_code_object(
      kernel_name_index_t kernel_index, const std::string &backend_kernel_name,
      backend_id b, Predicate &&object_selector, Constructor &&c) {

    return get_or_construct_code_object_impl(kernel_index, backend_kernel_name,
                                             b, object_selector, c);
  }

  template <class Constructor, class Predicate>
  const code_object *
  get_or_construct_code_object(kernel_name_index_t kernel_index,
                               const std::string &backend_kernel_name,
                               backend_id b, hcf_object_id source_object,
                               Predicate &&object_selector, Constructor &&c) {

    auto pred = [&](const code_object *obj) {
      if (obj->hcf_source() != source_object)
        return false;

      return object_selector(obj);
    };

    return get_or_construct_code_object(kernel_index, backend_kernel_name, b,
                                        pred, c);
  }

  // Only to be used within a Constructor passed to get_or_construct_code_object!
  template <class Constructor, class Predicate>
  const code_object *
  recursive_get_or_construct_code_object(kernel_name_index_t kernel_index,
                               const std::string &backend_kernel_name,
                               backend_id b, hcf_object_id source_object,
                               Predicate &&object_selector, Constructor &&c) {

    auto pred = [&](const code_object *obj) {
      if (obj->hcf_source() != source_object)
        return false;

      return object_selector(obj);
    };

    return recursive_get_or_construct_code_object(
        kernel_index, backend_kernel_name, b, pred, c);
  }

private:

  template<class Predicate>
  const code_object* get_code_object_impl(kernel_name_index_t kernel_index,
    const std::string& backend_kernel_name,
    backend_id b, Predicate&& object_selector) {
    
    assert(kernel_index < _kernel_names.size());

    auto& backend_code_objects = _kernel_code_objects[b];
    // Need an entry for every kernel
    if(backend_code_objects.size() != _kernel_names.size()) {
      backend_code_objects.resize(backend_code_objects.size());
    }

    auto& kernel_objects = backend_code_objects[kernel_index];

    // Check for best case: Desired code object exists
    // and is connected to the kernel
    for(code_object_index_t cidx : kernel_objects) {
      assert(cidx < _code_objects.size());
      if(object_selector(_code_objects[cidx].get()))
        return _code_objects[cidx].get();
    }

    // Check for second best case: Desired code object
    // exists, but it is not yet connected to the kernel.
    // We need to go through all the backend's code objects to check
    // if we can find it.
    for(code_object_index_t cidx = 0; cidx < _code_objects.size(); ++cidx) {
      const code_object* obj = _code_objects[cidx].get();
      if(obj->managing_backend() == b) {
        // Only investigate code objects as candidates where the predicate
        // says that they are relevant
        if(object_selector(obj)) {
          if(obj->contains(backend_kernel_name)) {
            // Connect this kernel to the object for future use
            kernel_objects.push_back(cidx);
            return obj;
          }
        }
      }
    }

    // Worst case: code object does not exist
    return nullptr;
  }

  template <class Constructor, class Predicate>
  const code_object *get_or_construct_code_object_impl(
      kernel_name_index_t kernel_index, const std::string &backend_kernel_name,
      backend_id b, Predicate &&object_selector, Constructor &&c) {

    code_object *obj = get_code_object_impl(kernel_index, backend_kernel_name,
                                            b, object_selector);
    if(obj)
      return obj;
    
    // We haven't found the requested object: Construct new code object
    code_object* new_obj = c();
    if(new_obj) {
      _code_objects.emplace_back(code_object_ptr{new_obj});
      code_object_index_t new_cidx = _code_objects.size() - 1;

      _kernel_code_objects[b][kernel_index].push_back(new_cidx);
    } else {
      register_error(
          __hipsycl_here(),
          error_info{"kernel_cache: code object creation has failed"});
    }

    return new_obj;
  }

  // These objects should only be written to during startup - they
  // are not thread-safe!
  std::vector<std::string> _kernel_names;
  std::unordered_map<std::string, kernel_name_index_t> _kernel_index_map;
  std::unordered_map<hcf_object_id, common::hcf_container> _hcf_objects;

  // These objects are thread-safe and can be written and read from multiple
  // threads at runtime if the mutex is locked.
  // Maps kernel_name_index_t to code objects
  common::small_map<rt::backend_id, std::vector<std::vector<code_object_index_t>>> _kernel_code_objects;

  std::vector<code_object_ptr> _code_objects;

  kernel_cache() = default;

  mutable std::mutex _mutex;
};

namespace detail {

template<class T>
struct kernel_registrator {
  kernel_registrator() { kernel_cache::get().register_kernel<T>(); }
};

template<class KernelT>
struct static_kernel_registration {
  static kernel_registrator<KernelT> init;
};

template<class KernelT>
kernel_registrator<KernelT> static_kernel_registration<KernelT>::init = {};

} // detail


}
}

#endif
