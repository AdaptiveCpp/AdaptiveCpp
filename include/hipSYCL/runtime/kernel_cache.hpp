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
#include "hipSYCL/glue/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"

#ifndef HIPSYCL_RT_KERNEL_CACHE_HPP
#define HIPSYCL_RT_KERNEL_CACHE_HPP

namespace hipsycl {
namespace rt {

enum class compilation_flow {
  integrated_multipass,
  explicit_multipass,
  sscp
};

enum class code_format {
  ptx,
  spirv,
  native_isa
};

enum class code_object_state {
  invalid,
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
  virtual compilation_flow source_compilation_flow() const = 0;

  /// Returns the kernel configuration id. This can e.g. be used
  /// to distinguish kernels with different specialization constant values /
  /// S2 IR constant values.
  virtual glue::kernel_configuration::id_type configuration_id() const {
    return glue::kernel_configuration::id_type{};
  }
  
  // Do we really need this? Cannot be implemented on all backends,
  // and may return empty vector in this case. Maybe better to not
  // have as part of the general interface?
  virtual std::vector<std::string> supported_backend_kernel_names() const = 0;

  virtual bool contains(const std::string& backend_kernel_name) const = 0;
};

// kernel information stored in HCF kernels as e.g. generated by the 
// SSCP compilation flow
class hcf_kernel_info {
public:
  hcf_kernel_info() = default;
  hcf_kernel_info(hcf_object_id id,
                  const common::hcf_container::node *kernel_node);

  std::size_t get_num_parameters() const;

  enum argument_type {
    pointer,
    other
  };

  enum annotation_type {
    specialized
  };

  std::size_t get_argument_offset(std::size_t i) const;
  std::size_t get_argument_size(std::size_t i) const;
  std::size_t get_original_argument_index(std::size_t i) const;
  argument_type get_argument_type(std::size_t i) const;
  const std::vector<std::string>& get_string_annotations(std::size_t i) const;
  const std::vector<annotation_type>& get_known_annotations(std::size_t i) const;

  bool is_valid() const;

  const std::vector<std::string> &get_images_containing_kernel() const;
  hcf_object_id get_hcf_object_id() const;

  const std::vector<glue::kernel_build_flag>& get_compilation_flags() const;
  const std::vector<std::pair<glue::kernel_build_option, std::string>> &
  get_compilation_options() const;

private:
  // We have one entry per kernel parameter for these
  std::vector<std::size_t> _arg_offsets;
  std::vector<std::size_t> _arg_sizes;
  std::vector<std::size_t> _original_arg_indices;
  std::vector<argument_type> _arg_types;
  std::vector<std::vector<std::string>> _string_annotations;
  std::vector<std::vector<annotation_type>> _known_annotations;

  std::vector<std::string> _image_providers;
  
  std::vector<glue::kernel_build_flag> _compilation_flags;
  std::vector<std::pair<glue::kernel_build_option, std::string>>
      _compilation_options;

  hcf_object_id _id;
  bool _parsing_successful = false;
};

// device image information as stored e.g. by the SSCP compilation flow.
class hcf_image_info {
public:
  hcf_image_info() = default;
  hcf_image_info(const common::hcf_container *hcf,
                 const common::hcf_container::node *image_node);

  const std::vector<std::string>& get_contained_kernels() const;
  // TODO: Maybe better return an enum of allowed formats/variants?
  const std::string& get_format() const;
  const std::string& get_variant() const;

  bool is_valid() const;
private:
  std::vector<std::string> _contained_kernels;
  std::string _format;
  std::string _variant;
  bool _parsing_successful = false;
};

// Stores all HCF data, and also extracts information for data
// in the SSCP format.
//
// This class is thread-safe.
class hcf_cache {
public:
  static hcf_cache& get();

  const common::hcf_container* get_hcf(hcf_object_id obj) const;
  
  hcf_object_id register_hcf_object(const common::hcf_container& obj);
  void unregister_hcf_object(hcf_object_id id);

  struct device_image_id {
    hcf_object_id hcf_id;
    const common::hcf_container::node* image_node;
  };

  using symbol_resolver_list = std::vector<device_image_id>;
  
  template<class Handler>
  void symbol_lookup(const std::vector<std::string>& names, Handler&& h) const {
    std::lock_guard<std::mutex> lock{_mutex};

    for(const auto& symbol_name : names) {
      HIPSYCL_DEBUG_INFO << "hcf_cache: Looking up symbol " << symbol_name
                         << std::endl;
      auto it = _exported_symbol_providers.find(symbol_name);
      if(it == _exported_symbol_providers.end()) {
        HIPSYCL_DEBUG_INFO << "hcf_cache: (Symbol not found)\n";
        h(symbol_name, {});
      } else {
        HIPSYCL_DEBUG_INFO << "hcf_cache: Symbol found\n";
        h(symbol_name, it->second);
      }
    }
  }

  const hcf_kernel_info *get_kernel_info(hcf_object_id obj,
                                         const std::string &kernel_name) const;

  const hcf_image_info *get_image_info(hcf_object_id obj,
                                       const std::string &image_name) const;

private:
  hcf_cache() = default;

  std::unordered_map<hcf_object_id, std::unique_ptr<common::hcf_container>>
      _hcf_objects;
  std::unordered_map<std::string, symbol_resolver_list> _exported_symbol_providers;

    
  struct stable_running_pair_hash {
    size_t operator()(const std::pair<hcf_object_id, std::string> &p) const
    {
      common::stable_running_hash h;
      h(reinterpret_cast<const void*>(&p.first), sizeof(hcf_object_id));
      h(static_cast<const void*>(p.second.c_str()), p.second.size());
      return h.get_current_hash();
    }
  };

  std::unordered_map<std::pair<hcf_object_id, std::string>,
                     std::unique_ptr<hcf_kernel_info>, stable_running_pair_hash>
      _hcf_kernel_info;
  std::unordered_map<std::pair<hcf_object_id, std::string>,
                     std::unique_ptr<hcf_image_info>, stable_running_pair_hash>
      _hcf_image_info;

  mutable std::mutex _mutex;
};

class kernel_cache {
public:
  using code_object_id = glue::kernel_configuration::id_type;
  using code_object_ptr = std::unique_ptr<const code_object>;

  static std::shared_ptr<kernel_cache> get();

  template<class KernelT>
  void register_kernel() {
    // This function is not needed in the current implementation, but it might
    // be useful in the future.
    std::string name = typeid(KernelT).name();
    HIPSYCL_DEBUG_INFO << "kernel_cache: Registering kernel " << name << std::endl;
  }

  /// Retrieve object for provided code object id, or nullptr
  /// if not found.
  const code_object* get_code_object(code_object_id id) const;

  /// Obtain or construct code objects. This is only for code objects
  /// that do not need to rely on our persistent kernel cache for JIT compilation
  /// results. The provided code object id is allowed to rely on values which might
  /// change between application runs.
  template <class Constructor>
  const code_object *get_or_construct_code_object(code_object_id id,
                                                  Constructor &&c) {
    std::lock_guard<std::mutex> lock{_mutex};
    return get_or_construct_code_object_impl(id, c);
  }

  /// Obtain or construct code objects. This is for code objects
  /// which rely on AdaptiveCpp-handled JIT compilation.
  /// In order to implement optimizations such as persistent on-disk kernel cache,
  /// we need to have explicit access to the JIT-compiled binary and distinguish
  /// the act of JIT compilation from constructing the backend code objects (e.g. CUmodule).
  ///
  /// This is why this function has two factory function arguments, and two ids:
  /// \c id_of_binary: A unique id of the binary. This value should only include configuration
  /// that is relevant for the jit-compiled code. It should not depend on any values
  /// that might vary between application runs (e.g. cl_context), because the binary
  /// might be persistently cached on-disk.
  /// \c id_of_code_object: The full id of the backend code object that the user wants to obtain.
  /// This id may depend on values which vary between application runs, such as cl_context.
  /// \c j Has signature bool(std::string&). Will be invoked when JIT compilation is triggered, and
  /// is expected to carry out JIT compilation.
  /// Should return true if the compilation was successful. The binary output of JIT compilation
  /// should be stored in the string reference.
  /// \c c Is expected to turn the JIT-compiled binary into a code_object*. Has signature
  /// code_object*(const std::string&). It is expected to return nullptr on error. The JIT-compiled
  /// binary will be passed in as string reference.
  template <class CodeObjectConstructor, class JitCompiler>
  const code_object *get_or_construct_jit_code_object(code_object_id id_of_code_object,
                                                      code_object_id id_of_binary,
                                                      JitCompiler &&jit_compile,
                                                      CodeObjectConstructor &&c) {
    if(auto* code_object = get_code_object(id_of_code_object)) {
      HIPSYCL_DEBUG_INFO << "kernel_cache: Cache hit for id "
                         << glue::kernel_configuration::to_string(id_of_code_object) << "\n";
      return code_object;
    }
    HIPSYCL_DEBUG_INFO << "kernel_cache: Cache MISS for id "
                      << glue::kernel_configuration::to_string(id_of_code_object) << "\n";
    
    std::string compiled_binary;
    // TODO: We might want to allow JIT compilation in parallel at some point
    std::lock_guard<std::mutex> lock{_mutex};

    if(!persistent_cache_lookup(id_of_binary, compiled_binary)){
      if(!jit_compile(compiled_binary))
        return nullptr;

      if(_is_first_jit_compilation) {
        _is_first_jit_compilation = false;
        HIPSYCL_DEBUG_WARNING
            << "kernel_cache: This application run has resulted in new "
               "binaries being JIT-compiled. This indicates that the runtime "
               "optimization process has not yet reached peak performance. You "
               "may want to run the application again until this warning no "
               "longer appears to achieve optimal performance."
            << std::endl;
      }
      persistent_cache_store(id_of_binary, compiled_binary);
    }
    
    const code_object* new_object = c(compiled_binary);
    if(new_object)
      _code_objects[id_of_code_object] = code_object_ptr{new_object};
    
    return new_object;
  }

  // Unload entire cache and release resources to prepare runtime shutdown.
  void unload();

  // Stitches together the persisten cache path with the id of the binary to a unique path.
  static std::string get_persistent_cache_file(code_object_id id_of_binary);
private:
  bool persistent_cache_lookup(code_object_id id_of_binary, std::string& out) const;
  void persistent_cache_store(code_object_id id_of_binary, const std::string& data) const;
  
  const code_object* get_code_object_impl(code_object_id id) const;

  template <class Constructor>
  const code_object *get_or_construct_code_object_impl(code_object_id id,
                                                  Constructor &&c) {
    auto* existing_code_object = get_code_object_impl(id);
    if(existing_code_object) {
      HIPSYCL_DEBUG_INFO << "kernel_cache: Cache hit for id "
                         << glue::kernel_configuration::to_string(id) << "\n";
      return existing_code_object;
    }
    HIPSYCL_DEBUG_INFO << "kernel_cache: Cache MISS for id "
                      << glue::kernel_configuration::to_string(id) << "\n";

    const code_object* new_object = c();
    if(new_object) {
      _code_objects[id] = code_object_ptr{new_object};
    }
    return new_object;
  }

  mutable std::mutex _mutex;

  std::unordered_map<code_object_id, code_object_ptr, glue::kernel_id_hash>
      _code_objects;
  
  bool _is_first_jit_compilation = true;
};

namespace detail {

template<class T>
struct kernel_registrator {
  kernel_registrator() { kernel_cache::get()->register_kernel<T>(); }
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
