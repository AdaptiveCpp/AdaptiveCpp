/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_GLUE_JIT_HPP
#define HIPSYCL_GLUE_JIT_HPP

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/small_map.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"
#include "hipSYCL/runtime/application.hpp"
#include <vector>
#include <atomic>
#include <fstream>

namespace hipsycl {
namespace glue {
namespace jit {

class default_llvm_image_selector {
public:
  std::string operator()(const common::hcf_container::node* kernel_node) const {
    assert(kernel_node);
    if(auto sn = kernel_node->get_subnode("format.llvm-ir")) {
      // Try all variants
      for(const auto& variant : sn->get_subnodes()) {
        if(auto vn = sn->get_subnode(variant)) {
          const std::string* res = vn->get_value("image-provider");
          if(res)
            return *res;
        }
      }
    }
    return std::string {};
    
  }

};

// Satisfies the image selector concept, but also
// finds all kernels associated with the selected image.
template<class ImageSelector>
class image_selector_and_kernel_list_extractor {
public:
  image_selector_and_kernel_list_extractor(
      ImageSelector &&sel, const std::string& kernel_name, 
      std::vector<std::string> *kernel_names_out,
      const common::hcf_container *hcf) {
    
    if(hcf && hcf->root_node()) {
      auto kernels = hcf->root_node()->get_subnode("kernels");
      if(!kernels) {
        HIPSYCL_DEBUG_WARNING
            << "image_selector_and_kernel_list_extractor: HCF does not contain "
               "'kernels' subnode, cannot select image"
            << "\n";
        return;
      }
      
      if(auto kernel_node = kernels->get_subnode(kernel_name)) {
        _selected_image = sel(kernel_node);
      } else {
        HIPSYCL_DEBUG_WARNING
            << "image_selector_and_kernel_list_extractor: HCF does not contain "
               "subnode for kernel, cannot select image"
            << "\n";
        return;
      }

      if(kernel_names_out) {
        *kernel_names_out = find_kernels(hcf, _selected_image);
      }
    }
  }

  std::string operator()() const {
    return _selected_image;
  }

private:
  std::vector<std::string> find_kernels(const common::hcf_container *hcf,
                                        const std::string &image_name) const {
    if(hcf && hcf->root_node()) {
      
      std::vector<std::string> result;

      auto* kernels = hcf->root_node()->get_subnode("kernels");
      if(kernels) {
        for(const auto& kernel_name : kernels->get_subnodes()) {
          auto* current_kernel = kernels->get_subnode(kernel_name);
          for(const auto& format : current_kernel->get_subnodes()) {
            auto* f = current_kernel->get_subnode(format);
            for(const auto& variant : f->get_subnodes()) {
              const std::string* provider = f->get_subnode(variant)->get_value("image-provider");
              if(provider && (*provider == image_name)) {
                result.push_back(kernel_name);
              }
            }
          }
        }
      }

      return result;
    }

    return {};
  }

  std::string _selected_image;
};

using symbol_list_t = compiler::LLVMToBackendTranslator::SymbolListType;

class runtime_linker {
  
public:
  using resolver = compiler::LLVMToBackendTranslator::ExternalSymbolResolver;
  using llvm_module_id = resolver::LLVMModuleId;


  runtime_linker(compiler::LLVMToBackendTranslator *translator,
                            const symbol_list_t &imported_symbol_names) {

    auto symbol_mapper = [this](const symbol_list_t& sl){ return this->map_smybols(sl); };

    auto bitcode_retriever = [this](llvm_module_id id,
                                    symbol_list_t &imported_symbols) {
      return this->retrieve_bitcode(id, imported_symbols);
    };

    translator->provideExternalSymbolResolver(
        resolver{symbol_mapper, bitcode_retriever, imported_symbol_names});
  }


private:

  std::vector<llvm_module_id> map_smybols(const symbol_list_t& sym_list) {
    std::vector<llvm_module_id> ir_modules_to_link;

    auto candidate_selector = [&, this](const std::string &symbol_name,
            const rt::hcf_cache::symbol_resolver_list &images) {
      for (const auto &img : images) {
        // Always attempt to link with global LLVM IR for now
        if (img.image_node->node_id == "llvm-ir.global") {
          _image_node_to_hcf_map[img.image_node] = img.hcf_id;
          ir_modules_to_link.push_back(
              reinterpret_cast<llvm_module_id>(img.image_node));
        } else {

          HIPSYCL_DEBUG_INFO << "jit::setup_linking: Discarding image "
                            << img.image_node->node_id << " @"
                            << img.image_node << " from HCF " << img.hcf_id
                            << "\n";
        }
      }
    };

    rt::hcf_cache::get().symbol_lookup(
        sym_list, candidate_selector);

    return ir_modules_to_link;
  }

  std::string retrieve_bitcode(llvm_module_id id, symbol_list_t& imported_symbols) const {

    const auto* hcf_image_node = reinterpret_cast<common::hcf_container::node*>(id);

    assert(_image_node_to_hcf_map.contains(hcf_image_node));

    auto v = _image_node_to_hcf_map.find(hcf_image_node);
    
    if(v == _image_node_to_hcf_map.end())
      return {};
    
    rt::hcf_object_id hcf_id = v->second;
    imported_symbols = hcf_image_node->get_as_list("imported-symbols");

    std::string bitcode;
    rt::hcf_cache::get().get_hcf(hcf_id)->get_binary_attachment(hcf_image_node, bitcode);

    return bitcode;
  }

  // This is used to map images to the owning HCF object ids.
  common::small_map<const common::hcf_container::node *, rt::hcf_object_id>
        _image_node_to_hcf_map;

};

inline rt::result compile(compiler::LLVMToBackendTranslator *translator,
                          const std::string &source,
                          const glue::kernel_configuration &config,
                          const symbol_list_t& imported_symbol_names,
                          std::string &output) {

  assert(translator);

  runtime_linker linker {translator, imported_symbol_names};

  // Apply configuration
  for(const auto& entry : config.entries()) {
    translator->setS2IRConstant(entry.get_name(), entry.get_data_buffer());
  }

  // Transform code
  if(!translator->fullTransformation(source, output)) {
    // In case of failure, if a dump directory for IR is set,
    // dump the IR
    auto failure_dump_directory =
        rt::application::get_settings()
            .get<rt::setting::sscp_failed_ir_dump_directory>();
            
    if(!failure_dump_directory.empty()) {
      static std::atomic<std::size_t> failure_index = 0;
      std::string filename = common::filesystem::join_path(
          failure_dump_directory,
          "failed_ir_" + std::to_string(failure_index) + ".bc");
      
      std::ofstream out{filename.c_str(), std::ios::trunc|std::ios::binary};
      if(out.is_open()) {
        const std::string& failed_ir = translator->getFailedIR();
        out.write(failed_ir.c_str(), failed_ir.size());
      }

      ++failure_index;
    }
    
    return rt::make_error(__hipsycl_here(),
                      rt::error_info{"jit::compile: Encountered errors:\n" +
                                 translator->getErrorLogAsString()});
  }

  return rt::make_success();
}


// ProviderSelector is of signature std::string (const hcf_container::node*) and
// is supposed to select one of the kernel image providers for compilation. The
// node argument will be set to the kernel node in the HCF, such that the
// subnodes are the list of available image formats.
template<class ProviderSelector>
inline rt::result compile(compiler::LLVMToBackendTranslator* translator,
                          const common::hcf_container* hcf,
                          ProviderSelector&& provider_selector,
                          const glue::kernel_configuration &config,
                          std::string &output) {
  assert(hcf);
  assert(hcf->root_node());

 
  std::string selected_kernel_provider = provider_selector();
  
  if(selected_kernel_provider.empty()) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{
            "jit::compile: kernel provider selector did not select kernel."});
  }

  auto images_node = hcf->root_node()->get_subnode("images");
  if(!images_node) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{
            "jit::compile: Invalid HCF, no node named 'images' was found"});
  }

  auto target_image_node = images_node->get_subnode(selected_kernel_provider);
  if(!target_image_node) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{"jit::compile: Image " + selected_kernel_provider +
                       " referenced in kernel node, but not defined in HCF"});
  }

  if(!target_image_node->has_binary_data_attached()) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{"jit::compile: Image " + selected_kernel_provider +
                       " was defined in HCF without data"});
  }
  std::string source;
  if(!hcf->get_binary_attachment(target_image_node, source)) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{
            "jit::compile: Could not extract binary data for HCF image " +
            selected_kernel_provider});
  }

  symbol_list_t imported_symbol_names =
      target_image_node->get_as_list("imported-symbols");

  return compile(translator, source, config, imported_symbol_names, output);
}

template<class ProviderSelector>
inline rt::result compile(compiler::LLVMToBackendTranslator* translator,
                          rt::hcf_object_id hcf_object,
                          ProviderSelector&& provider_selector,
                          const glue::kernel_configuration &config,
                          std::string &output) {
  const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
  if(!hcf) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{"jit::compile: Could not obtain HCF object"});
  }

  return compile(translator, hcf, provider_selector, config,
                 output);
}

}
}
}

#endif
