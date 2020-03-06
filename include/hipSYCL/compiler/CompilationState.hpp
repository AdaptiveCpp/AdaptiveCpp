/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_COMPILATION_STATE_HPP
#define HIPSYCL_COMPILATION_STATE_HPP

#include <string>
#include <unordered_set>

namespace hipsycl {
namespace compiler {

class ASTPassState
{
  std::unordered_set<std::string> ImplicitlyMarkedHostDeviceFunctions;
  std::unordered_set<std::string> ExplicitDeviceFunctions;
  std::unordered_set<std::string> KernelFunctions;
  bool IsDeviceCompilation;
public:
  ASTPassState()
  : IsDeviceCompilation{false}
  {}

  bool isDeviceCompilation() const
  {
    return IsDeviceCompilation;
  }

  void setDeviceCompilation(bool IsDevice)
  {
    this->IsDeviceCompilation = IsDevice;
  }

  void addImplicitHostDeviceFunction(const std::string& Name)
  {
    ImplicitlyMarkedHostDeviceFunctions.insert(Name);
  }

  void addKernelFunction(const std::string& Name)
  {
    KernelFunctions.insert(Name);
  }

  void addExplicitDeviceFunction(const std::string& Name)
  {
    ExplicitDeviceFunctions.insert(Name);
  }

  bool isImplicitlyHostDevice(const std::string& FunctionName) const
  {
    return ImplicitlyMarkedHostDeviceFunctions.find(FunctionName) 
      != ImplicitlyMarkedHostDeviceFunctions.end();
  }

  bool isExplicitlyDevice(const std::string& FunctionName) const
  {
    return ExplicitDeviceFunctions.find(FunctionName)
      != ExplicitDeviceFunctions.end();
  }

  bool isKernel(const std::string& FunctionName) const
  {
    return KernelFunctions.find(FunctionName)
      != KernelFunctions.end();
  }

};

class CompilationStateManager
{
public:
  static CompilationStateManager& get()
  {
    static CompilationStateManager m;
    return m;
  }

  void reset()
  {
    ASTState = ASTPassState();
  }

  static ASTPassState &getASTPassState() { return get().ASTState; }

private:
  CompilationStateManager() = default;
  ASTPassState ASTState;
};

} // namespace compiler
} // namespace hipsycl

#endif
