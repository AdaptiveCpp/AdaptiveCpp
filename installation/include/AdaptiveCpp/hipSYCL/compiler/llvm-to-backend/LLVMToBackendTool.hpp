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
  std::cout
      << "Usage: llvm-to-<backend> [--ir] [--build-opt <BackendBuildOptionName>=<Value>] "
         "[--build-flag <BackendBuildFlagName>] <HCF inputfile> <outputfile> <device-image-name>"
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

inline bool splitBuildArg(const std::string& S, std::string& ArgNameOut, std::string& ArgValueOut) {
  std::size_t pos = S.find("=");
  if(pos == std::string::npos || pos == S.size() - 1)
    return false;

  ArgNameOut = S.substr(0, pos);
  ArgValueOut = S.substr(pos+1);

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

  bool PartialTranslation = false;
  std::vector<std::string> BuildFlags;
  std::vector<std::pair<std::string, std::string>> BuildOptions;

  int GeneralArgsStart = 1;
  bool GeneralArgEncountered = false;

  while(GeneralArgsStart < argc && !GeneralArgEncountered) {
    if(argv[GeneralArgsStart] == std::string{"--ir"}) {
      PartialTranslation = true;
    } else if (argv[GeneralArgsStart] == std::string{"--build-opt"}) {
      if(GeneralArgsStart + 1 < argc) {
        ++GeneralArgsStart;
      } else {
        help();
        return -1;
      }

      std::string ArgName, ArgValue;
      if(!splitBuildArg(argv[GeneralArgsStart], ArgName, ArgValue)) {
        help();
        return -1;
      }

      BuildOptions.push_back(std::make_pair(ArgName, ArgValue));
    } else if (argv[GeneralArgsStart] == std::string{"--build-flag"}) {
      if(GeneralArgsStart + 1 < argc) {
        ++GeneralArgsStart;
      } else {
        help();
        return -1;
      }

      BuildFlags.push_back(argv[GeneralArgsStart]);
    } else {
      GeneralArgEncountered = true;
    }

    if(!GeneralArgEncountered)
      ++GeneralArgsStart;
  }
  if(argv[GeneralArgsStart] == std::string{"--ir"}) {
    PartialTranslation = true;
    ++GeneralArgsStart;
  }

  if(GeneralArgsStart+3 > argc) {
    help();
    return -1;
  }

  std::string InputFile = argv[GeneralArgsStart];
  std::string OutputFile = argv[GeneralArgsStart+1];
  std::string ImageName = argv[GeneralArgsStart+2];

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

  for(const auto& F : BuildFlags) {
    Translator->setBuildFlag(F);
  }
  for(const auto& O : BuildOptions) {
    Translator->setBuildOption(O.first, O.second);
  }

  bool Result = false;
  if(!PartialTranslation) {
    Result = Translator->fullTransformation(IR, Output);
  } else {
    Result = Translator->partialTransformation(IR, Output);
  }

  if(!Result){
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