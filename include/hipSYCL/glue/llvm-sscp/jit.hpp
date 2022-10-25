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
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"
#include <vector>

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
      ImageSelector &&sel, std::vector<std::string> *kernel_names_out,
      const common::hcf_container *hcf)
      : _selector{sel}, _kernel_names{kernel_names_out}, _hcf{hcf} {}

  std::string operator()(const common::hcf_container::node* kernel_node) const {
    std::string image_name = _selector(kernel_node);

    if(!image_name.empty() && _kernel_names && _hcf) {
      *_kernel_names = find_kernels(_hcf, image_name);
    }

    return image_name;
  }

private:

  std::vector<std::string> find_kernels(const common::hcf_container *hcf,
                                        const std::string &image_name) {
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

  ImageSelector _selector;
  std::vector<std::string>* _kernel_names;
  const common::hcf_container* _hcf;
};

inline rt::result compile(compiler::LLVMToBackendTranslator *translator,
                          const std::string &source,
                          const glue::kernel_configuration &config,
                          std::string &output) {

  assert(translator);

  // TODO Link with SYCL_EXTERNAL symbols

  // Apply configuration
  for(const auto& entry : config.entries()) {
    translator->setS2IRConstant(entry.get_name(), entry.get_data_buffer());
  }

  if(!translator->fullTransformation(source, output)) {
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
inline rt::result compile(compiler::LLVMToBackendTranslator *translator,
                          common::hcf_container* hcf,
                          const std::string& kernel_name,
                          ProviderSelector&& provider_selector,
                          const glue::kernel_configuration &config,
                          std::string &output) {
  assert(hcf);
  assert(hcf->root_node());

  auto* kernel_list_node = hcf->root_node()->get_subnode("kernels");

  if(!kernel_list_node) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{
            "jit::compile: Invalid HCF, no node named 'kernels' was found"});
  }

  auto kernel_node = kernel_list_node->get_subnode(kernel_name);

  if(!kernel_node) {
    return rt::make_error(__hipsycl_here(),
                          rt::error_info{"jit::compile: HCF node for kernel " +
                                         kernel_name + " not found."});
  }

  std::string selected_kernel_provider = provider_selector(kernel_node);
  
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

  return compile(translator, source, config, output);
}

template<class ProviderSelector>
inline rt::result compile(compiler::LLVMToBackendTranslator *translator,
                          rt::hcf_object_id hcf_object,
                          const std::string& kernel_name,
                          ProviderSelector&& provider_selector,
                          const glue::kernel_configuration &config,
                          std::string &output) {
  const common::hcf_container* hcf = rt::kernel_cache::get().get_hcf(hcf_object);
  if(!hcf) {
    return rt::make_error(
        __hipsycl_here(),
        rt::error_info{"jit::compile: Could not obtain HCF object"});
  }

  return compile(translator, hcf, kernel_name, provider_selector, config,
                 output);
}

}
}
}

#endif
