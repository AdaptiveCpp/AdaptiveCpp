/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2021 Aksel Alpay
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

#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "hipSYCL/runtime/cuda/cuda_module.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

namespace {


void trim_left(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

void trim_right_space_and_parenthesis(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch) && (ch != '(');
    }).base(), s.end());
}

class cuda_module_cache {
public:
  static cuda_module_cache& get() {
    static cuda_module_cache instance;
    return instance;
  }

  ~cuda_module_cache() {
    for (std::size_t device = 0; device < _loaded_modules.size(); ++device) {
      auto& per_device_cache = _loaded_modules[device];
      for (auto mod : per_device_cache) {
        CUmod_st* cuda_mod = mod.second;
        if(cuda_mod) {
          cuda_device_manager::get().activate_device(device);

          auto err = cuModuleUnload(cuda_mod);

          if (err != CUDA_SUCCESS) {
            register_error(__hipsycl_here(),
                          error_info{"cuda_module_cache: could not unload module",
                                    error_code{"CU", static_cast<int>(err)}});
          }
          _loaded_modules[device][mod.first] = nullptr;
        }
      }
    }
  }

  template <class Generator>
  result create_or_retrieve_module(cuda_module_id_t module_id,
                                   const std::string &target, cuda_module *&out,
                                   Generator gen) {
    for(auto& mod : _source_modules) {
      if(mod.get_id() == module_id && mod.get_target() == target) {
        out = &mod;
        return make_success();
      }
    }

    _source_modules.emplace_back(gen(module_id, target));

    out = &(_source_modules.back());
    return make_success();
  }

  template <class Generator>
  result create_or_retrieve_backend_module(rt::device_id dev,
                                           const cuda_module &module,
                                           CUmod_st *&out, Generator gen) {
    assert(dev.get_id() < _loaded_modules.size());

    auto& per_device_cache = _loaded_modules[dev.get_id()];
    auto it = per_device_cache.find(module.get_id());

    if(it != per_device_cache.end()) {
      out = it->second;
      return make_success();
    } else {
      CUmod_st* mod = nullptr;
      result err = gen(dev, module, mod);
      
      if(!err.is_success())
        return err;
      
      per_device_cache[module.get_id()] = mod;
      out = mod;
      
      return make_success();
    }
  }

private:

  cuda_module_cache() {
    int num_devices = 0;
    auto err = cudaGetDeviceCount(&num_devices);

    if(err != cudaSuccess) {
      print_warning(__hipsycl_here(),
        error_info{"cuda_module_cache: could not obtain number of devices",
          error_code{"CU", static_cast<int>(err)}});
    }

    if(num_devices > 0)
      _loaded_modules.resize(num_devices);
  }
  

  std::vector<cuda_module> _source_modules;
  std::vector<std::unordered_map<cuda_module_id_t, CUmod_st*>> _loaded_modules;
};

}

cuda_module::cuda_module(cuda_module_id_t module_id, const std::string &target,
                         const std::string &code_content)
    : _id{module_id}, _target{target}, _content{code_content} {

  std::istringstream code_stream(code_content);
  std::string line;

  while (std::getline(code_stream, line)) {

    const std::string kernel_identifier = ".visible .entry";
    auto pos = line.find(kernel_identifier);

    if (pos != std::string::npos) {
      line = line.substr(pos+kernel_identifier.size());
      trim_left(line);
      trim_right_space_and_parenthesis(line);
      HIPSYCL_DEBUG_INFO << "Detected kernel in module: " << line << std::endl;
      _kernel_names.push_back(line);
    }
  }
}

const std::vector<std::string> &
cuda_module::get_kernel_names() const {
  return _kernel_names;
}

std::string cuda_module::get_content() const {
  return _content;
}

bool cuda_module::guess_kernel_name(const std::string &kernel_group_name,
                                    const std::string &kernel_component_name,
                                    std::string &guessed_name) const {
  bool found = false;
  for (auto candidate : get_kernel_names()) {
    if (candidate.find(kernel_group_name) != std::string::npos &&
        candidate.find(kernel_component_name) != std::string::npos) {

      if (found) {
        HIPSYCL_DEBUG_WARNING
            << "Encountered multiple candidates for kernels from group "
            << kernel_group_name << " with component: " << kernel_component_name
            << ": " << candidate << std::endl;
        HIPSYCL_DEBUG_WARNING << "Keeping initial guess: " << guessed_name
                              << std::endl;
      } else {
        guessed_name = candidate;
        found = true;
      }
    }
  }
  
  return found;
}

cuda_module_id_t cuda_module::get_id() const { return _id; }

const std::string &cuda_module::get_target() const { return _target; }

cuda_module_manager::cuda_module_manager(std::size_t num_devices){}

cuda_module_manager::~cuda_module_manager() {}

const cuda_module &
cuda_module_manager::obtain_module(cuda_module_id_t id,
                                   const std::string &target,
                                   const std::string &content) {
  cuda_module* mod = nullptr;
  cuda_module_cache::get().create_or_retrieve_module(
      id, target, mod, [&content](cuda_module_id_t id, const std::string &target) {
        return cuda_module{id, target, content};
      });
  assert(mod);
  return *mod;
}

result cuda_module_manager::load(rt::device_id dev, const cuda_module &module,
                                 CUmod_st *&out) {
  
  assert(dev.get_backend() == backend_id::cuda);
  
  result err = cuda_module_cache::get().create_or_retrieve_backend_module(dev, module, out,
    [](rt::device_id dev, const cuda_module& mod, CUmod_st*& out){
    
    cuda_device_manager::get().activate_device(dev.get_id());
    // This guarantees that the CUDA runtime API initializes the CUDA
    // context on that device. This is important for the subsequent driver
    // API calls which assume that CUDA context has been created.
    cudaFree(0);
    CUmod_st* m = nullptr;
    auto err = cuModuleLoadDataEx(
        &m, static_cast<void *>(const_cast<char *>(mod.get_content().c_str())),
        0, nullptr, nullptr);

    if (err != CUDA_SUCCESS) {
      return make_error(__hipsycl_here(),
                        error_info{"cuda_module_manager: could not load module",
                                  error_code{"CU", static_cast<int>(err)}});
    }
    
    out = m;

    return make_success();
  });
  
  return err;
}


} // namespace rt
} // namespace hipsycl
