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

#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>

namespace hipsycl {
namespace rt {

namespace {

template<class F>
void for_each_device_image(const common::hcf_container& hcf, F&& handler) {
  if(hcf.root_node()->has_subnode("images")) {
    auto* img_node = hcf.root_node()->get_subnode("images");
    auto image_names = img_node->get_subnodes();
    for(const auto& image_name : image_names) {
      handler(img_node->get_subnode(image_name));
    }
  }
}

template<class F>
void for_each_exported_symbol_list(const common::hcf_container& hcf, F&& handler) {
  for_each_device_image(hcf, [&](const common::hcf_container::node* image_node){
    auto exported_symbols = image_node->get_as_list("exported-symbols");
    handler(image_node, exported_symbols);
  });
}

}

kernel_cache& kernel_cache::get() {
  static kernel_cache c;
  return c;
}

hcf_cache& hcf_cache::get() {
  static hcf_cache c;
  return c;
}

hcf_object_id hcf_cache::register_hcf_object(const common::hcf_container &obj) {

  std::lock_guard<std::mutex> lock{_mutex};

  if (!obj.root_node()->has_key("object-id")) {
    HIPSYCL_DEBUG_ERROR
        << "hcf_cache: Invalid hcf object (missing object id)" << std::endl;
  }
  const std::string *data = obj.root_node()->get_value("object-id");

  assert(data);
  hcf_object_id id = std::stoull(*data);
  HIPSYCL_DEBUG_INFO << "hcf_cache: Registering HCF object " << id << "..." << std::endl;

  if (_hcf_objects.count(id) > 0) {
    HIPSYCL_DEBUG_ERROR
        << "hcf_cache: Detected hcf object id collision " << id
        << ", this should not happen. Some kernels might be unavailable."
        << std::endl;
  } else {
    common::hcf_container* stored_obj = new common::hcf_container{obj};
    _hcf_objects[id] = std::move(std::unique_ptr<common::hcf_container>{stored_obj});
    // Check if the HCF exports some symbols
    for_each_exported_symbol_list(
        // Don't use obj here, since we have copied it into the cache, and need
        // to ensure that the pointers to image nodes are stable
        *stored_obj,
        [&](const common::hcf_container::node *image_node,
            const std::vector<std::string> &exported_symbols) {

          for (const auto &symbol : exported_symbols) {

            _exported_symbol_providers[symbol].push_back(
                device_image_id{id, image_node});

            HIPSYCL_DEBUG_INFO << "hcf_cache: Symbol " << symbol
                               << " is registered as exported by object " << id
                               << " and image " << image_node->node_id
                               << " @" << image_node << std::endl;
          }
        });
  }

  std::string hcf_dump_dir =
      application::get_settings().get<setting::hcf_dump_directory>();
  if(!hcf_dump_dir.empty()) {
    std::string out_filename = hcf_dump_dir;

    if(out_filename.back() != '/' && out_filename.back() != '\\')
      out_filename += '/';
    
    out_filename += "hipsycl_object_"+std::to_string(id)+".hcf";

    std::ofstream out_file(out_filename.c_str(), std::ios::binary);
    if(!out_file.is_open()) {
      HIPSYCL_DEBUG_ERROR << "Could not open file " << out_filename
                          << " for writing." << std::endl;

    } else {
      std::string hcf_data = obj.serialize();
      out_file.write(hcf_data.c_str(), hcf_data.size());
    }
  }

  return id;
}

void hcf_cache::unregister_hcf_object(hcf_object_id id) {
  std::lock_guard<std::mutex> lock{_mutex};

  auto it = _hcf_objects.find(id);
  if(it != _hcf_objects.end()) {
    // First remove the HCF object as a symbol provider for runtime linking and
    // symbol resolution. This ensures that it gets no longer selected
    // for symbol resolution.

    // 1. Go through each symbol list (all device images) exported by this
    // HCF file
    for_each_exported_symbol_list(
        *(it->second), [&](const common::hcf_container::node* image_node,
                 const std::vector<std::string> &exported_symbols) {
          // 2. Iterate over all symbols exported in this HCF
          for (const auto &symbol : exported_symbols) {
            // 3. Remove all references to this HCF in the symbol providers map  
            auto& symbol_providers = _exported_symbol_providers[symbol];
            symbol_providers.erase(
                std::remove_if(symbol_providers.begin(), symbol_providers.end(),
                               [&](const device_image_id &img) {
                                 return img.hcf_id == id;
                               }),
                symbol_providers.end());
          }
        });
    // Then we can remove the HCF itself
    _hcf_objects.erase(id);
  }
}

const common::hcf_container* hcf_cache::get_hcf(hcf_object_id obj) const {
  std::lock_guard<std::mutex> lock{_mutex};

  auto it = _hcf_objects.find(obj);
  if(it == _hcf_objects.end())
    return nullptr;
  return it->second.get();
}


const kernel_cache::kernel_name_index_t*
kernel_cache::get_global_kernel_index(const std::string &kernel_name) const {
  auto it = _kernel_index_map.find(kernel_name);
  if(it == _kernel_index_map.end())
    return nullptr;
  return &(it->second);
}


void kernel_cache::unload() {
  std::lock_guard<std::mutex> lock{_mutex};

  _kernel_code_objects.clear();
  _code_objects.clear();
}

} // rt
} // hipsycl