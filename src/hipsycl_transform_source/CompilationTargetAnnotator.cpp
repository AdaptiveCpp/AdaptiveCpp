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

#include "CL/sycl/detail/debug.hpp"

#include "CompilationTargetAnnotator.hpp"


using namespace clang;

namespace hipsycl {
namespace transform {

CompilationTargetAnnotatingASTVisitor::CompilationTargetAnnotatingASTVisitor(Rewriter& r)
  : _rewriter{r},
    _currentFunction{nullptr}
{}

bool CompilationTargetAnnotatingASTVisitor::VisitCallExpr(CallExpr *call)
{
  if(_currentFunction != nullptr)
  {
    FunctionDecl* callee = call->getDirectCallee();

    if(callee != nullptr)
      _callers[callee].push_back(_currentFunction);
  }
  return true;
}

bool CompilationTargetAnnotatingASTVisitor::VisitFunctionDecl(FunctionDecl *f)
{
  if(f->hasBody())
    _currentFunction = f;

  return true;
}

void CompilationTargetAnnotatingASTVisitor::addAnnotations()
{
  for(auto func : _callers)
  {
    bool isHost = false;
    bool isDevice = false;
    correctFunctionAnnotations(isHost, isDevice, func.first);
  }
}

bool
CompilationTargetAnnotatingASTVisitor::isHostFunction(clang::FunctionDecl* f) const
{
  return this->containsTargetAttribute(f, "host") ||
      (_isFunctionCorrectedHost.find(f) != _isFunctionCorrectedHost.end());
}

bool
CompilationTargetAnnotatingASTVisitor::isDeviceFunction(clang::FunctionDecl* f) const
{
  return this->containsTargetAttribute(f, "device") ||
      (_isFunctionCorrectedDevice.find(f) != _isFunctionCorrectedDevice.end());
}

bool
CompilationTargetAnnotatingASTVisitor::isKernelFunction(clang::FunctionDecl* f) const
{
  return this->containsTargetAttribute(f, "kernel");
}

bool
CompilationTargetAnnotatingASTVisitor::containsTargetAttribute(
    FunctionDecl* f,
    const std::string& targetString) const
{
  auto functionAttributes = f->getAttrs();
  for(Attr* currentAttrib : functionAttributes)
  {
    if(isa<TargetAttr>(currentAttrib))
    {
      TargetAttr* target = cast<TargetAttr>(currentAttrib);
      if(target->getFeaturesStr().str() == targetString)
        return true;
    }
  }
  return false;
}


bool CompilationTargetAnnotatingASTVisitor::canCallHostFunctions(
    clang::FunctionDecl* f) const
{
  return isHostFunction(f);
}

bool CompilationTargetAnnotatingASTVisitor::canCallDeviceFunctions(
    clang::FunctionDecl* f) const
{
  return isDeviceFunction(f) || isKernelFunction(f);
}


/// Determines if a function is called by host or device functions.
void
CompilationTargetAnnotatingASTVisitor::correctFunctionAnnotations(
    bool& isHost, bool& isDevice, clang::FunctionDecl* f)
{
  if(isKernelFunction(f))
    // Treat kernels as device functions since they execute on the device
    isDevice = true;
  else
  {
    if(_isFunctionProcessed[f])
    {
      isHost = isHostFunction(f);
      isDevice = isDeviceFunction(f);
    }
    else
    {
      // Set this already in the beginning to avoid getting stuck in
      // cycles in the call graph
      _isFunctionProcessed[f] = true;

      isHost = false;
      isDevice = false;

      auto callers = _callers[f];
#ifdef HIPSYCL_VERBOSE_DEBUG
      HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: "
                         << "call graph for "
                         << f->getNameAsString() << ": " << std::endl;
#endif
      for(clang::FunctionDecl* caller : callers)
      {
#ifdef HIPSYCL_VERBOSE_DEBUG
        HIPSYCL_DEBUG_INFO << "    called by "
                           << caller->getNameAsString() << std::endl;
#endif

        bool callerIsHostFunction = false;
        bool callerIsDeviceFunction = false;

        correctFunctionAnnotations(callerIsHostFunction, callerIsDeviceFunction, caller);

        isHost |= callerIsHostFunction;
        isDevice |= callerIsDeviceFunction;

        if(isHost && isDevice)
          break;
      }

      if(isHost)
        markAsHost(f);
      if(isDevice)
        markAsDevice(f);
    }
    // If device isn't explicitly marked as __host__ or __device__, treat it as
    // host
    if(!isDevice && !isHost)
    {
      isHost = true;
      markAsHost(f);
    }
  }
}

void CompilationTargetAnnotatingASTVisitor::markAs(
    clang::FunctionDecl* f,
    const std::string& annotation)
{
  for(FunctionDecl* currentDecl = f->getMostRecentDecl();
      currentDecl != nullptr;
      currentDecl = currentDecl->getPreviousDecl())
  {
    _rewriter.InsertText(currentDecl->getTypeSpecStartLoc(), annotation);
  }
}

void CompilationTargetAnnotatingASTVisitor::markAsHost(clang::FunctionDecl* f)
{
  if(isKernelFunction(f))
    return;

  if(!isHostFunction(f))
  {
#ifdef HIPSYCL_VERBOSE_DEBUG
    HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as __host__: "
                       << f->getNameAsString() << std::endl;
#endif
    _isFunctionCorrectedHost.insert(f);

    markAs(f, " __host__ ");

  }
}

void CompilationTargetAnnotatingASTVisitor::markAsDevice(clang::FunctionDecl* f)
{
  if(isKernelFunction(f))
    return;

  if(!isDeviceFunction(f))
  {
#ifdef HIPSYCL_VERBOSE_DEBUG
    HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as __device__: "
                       << f->getNameAsString() << std::endl;
#endif
    _isFunctionCorrectedDevice.insert(f);

    markAs(f, " __device__ ");
  }
}


}
}

