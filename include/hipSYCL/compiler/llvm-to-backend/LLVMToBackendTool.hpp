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

namespace hipsycl {
namespace compiler {

using TranslatorFactory =
    std::function<std::unique_ptr<hipsycl::compiler::LLVMToBackendTranslator>()>;

inline void help() {
  std::cout << "Usage: llvm-to-<backend> <inputfile> <outputfile>"
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

  if(argc < 3) {
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

  auto Translator = createTranslator();

  std::string Input, Output;
  if(!readFile(InputFile, Input)) {
    std::cout << "Could not open file: " << InputFile << std::endl;
  }

  if(!Translator->fullTransformation(Input, Output)){
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

#endif