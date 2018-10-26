/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include "HipsyclTransform.hpp"
#include "Attributes.hpp"
#include "../common/Paths.hpp"


using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;



int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, llvm::cl::GeneralCategory);

  ClangTool tool(op.getCompilations(), op.getSourcePathList());

  ArgumentsAdjuster adjuster =
      [](const CommandLineArguments& args,StringRef) -> CommandLineArguments
  {
    CommandLineArguments modifiedArgs =
        hipsycl::transform::Application::getCommandLineArgs().consumeHipsyclArgs(args);

    modifiedArgs.push_back("-D__global__="+hipsycl::transform::KernelAttribute::getString());
    modifiedArgs.push_back("-D__host__="+hipsycl::transform::HostAttribute::getString());
    modifiedArgs.push_back("-D__device__="+hipsycl::transform::DeviceAttribute::getString());
    //modifiedArgs.push_back("-D__constant__");
    //modifiedArgs.push_back("-D__shared__");

    modifiedArgs.push_back("-D__HIPSYCL_TRANSFORM__");

    modifiedArgs.push_back("-std=c++14");

    std::string clangIncludeDir = hipsycl::paths::getClangIncludePath();
    if(!clangIncludeDir.empty())
      modifiedArgs.push_back("-I"+clangIncludeDir);

    return modifiedArgs;
  };

  try
  {

    tool.appendArgumentsAdjuster(adjuster);

    using FrontendActionType = hipsycl::transform::HipsyclTransfromFrontendAction;
    return tool.run(newFrontendActionFactory<FrontendActionType>().get());
  }
  catch(std::exception& e)
  {
    std::cout << "hipsycl_transform_source error: " << e.what() << std::endl;
    return -1;
  }
}
