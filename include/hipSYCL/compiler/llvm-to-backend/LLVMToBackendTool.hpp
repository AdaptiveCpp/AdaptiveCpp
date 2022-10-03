/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#ifndef HIPSYCL_LLVM_TO_BACKEND_TOOL_HPP
#define HIPSYCL_LLVM_TO_BACKEND_TOOL_HPP

#include <ios>
#include <iostream>
#include <memory>
#include <fstream>
#include <functional>
#include "LLVMToBackend.hpp"
#include "hipSYCL/common/hcf_container.hpp"

namespace hipsycl {
namespace compiler {
namespace translation_tool {

inline bool getHcfKernelNames(const hipsycl::common::hcf_container &HCF,
                              std::vector<std::string> &KernelNamesOut) {
  if(!HCF.root_node())
    return false;
  
  auto* KernelsNode = HCF.root_node()->get_subnode("kernels");
  if(!KernelsNode)
    return false;

  KernelNamesOut = KernelsNode->get_subnodes();
  return true;
}

using TranslatorFactory =
    std::function<std::unique_ptr<hipsycl::compiler::LLVMToBackendTranslator>(
      const hipsycl::common::hcf_container& HCF
    )>;

inline void help() {
  std::cout << "Usage: llvm-to-<backend> <HCF inputfile> <outputfile> <device-image-name>"
            << std::endl;
}

inline bool readFile(const std::string& Filename, std::string& Out) {
  std::ifstream File{Filename, std::ios::binary|std::ios::ate};
  if(!File.is_open())
    return false;

  auto size = File.tellg();

  if (size == 0) {
      Out = std::string{};
      return true;
  }

  std::string result(size, '\0');

  File.seekg(0, std::ios::beg);
  File.read(result.data(), size);

  Out = result;

  return true;
}

inline bool writeFile(const std::string& Filename, const std::string& Data){
  std::ofstream File{Filename, std::ios::binary|std::ios::trunc};
  if(!File.is_open()){
    return false;
  }

  File.write(Data.data(), Data.size());
  File.close();
  return true;
}

inline int LLVMToBackendToolMain(int argc, char **argv, TranslatorFactory &&createTranslator) {

  if(argc < 4) {
    help();
    return -1;
  }

  for(int i = 1; i < argc; ++i) {
    if(argv[i] == std::string{"--help"}) {
      help();
      return 0;
    }
  }

  std::string InputFile = argv[1];
  std::string OutputFile = argv[2];
  std::string ImageName = argv[3];

  std::string HcfInput, Output;
  if(!readFile(InputFile, HcfInput)) {
    std::cout << "Could not open file: " << InputFile << std::endl;
  }

  common::hcf_container HCF{HcfInput};
  auto* ImgNode = HCF.root_node()->get_subnode("images");
  if(!ImgNode) {
    std::cout << "Invalid HCF: Could not find 'images' node" << std::endl;
    return -1;
  }
  ImgNode = ImgNode->get_subnode(ImageName);
  if(!ImgNode) {
    std::cout << "Invalid HCF: Could not find specified device image node" << std::endl;
    return -1;
  }
  if(!ImgNode->has_binary_data_attached()){
    std::cout << "Invalid HCF: Specified node has no data attached to it." << std::endl;
    return -1;
  }

  std::string IR;
  HCF.get_binary_attachment(ImgNode, IR);
  
  auto Translator = createTranslator(HCF);
  if(!Translator) {
    std::cout << "Could not construct backend translation object." << std::endl;
    return -1;
  }
  if(!Translator->fullTransformation(IR, Output)){
    std::cout << "Transformation failed." << std::endl;
    if(!Translator->getErrorLog().empty()) {
      std::cout << "The following issues have been encountered:" << std::endl;
      for(const auto& E : Translator->getErrorLog()){
        std::cout << E << std::endl;
      }
    }
    return -1;
  }
  
  if(!writeFile(OutputFile, Output)){
    std::cout << "Could not write output to file: " << OutputFile << std::endl;
  }

  return 0;
}

}
}
}

#endif