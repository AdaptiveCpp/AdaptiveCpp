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

#define HIPSYCL_VERBOSE_DEBUG

using namespace clang;

namespace hipsycl {
namespace transform {

CompilationTargetAnnotator::CompilationTargetAnnotator(Rewriter& r,
                                                       clang::CallGraph& callGraph)
  : _rewriter{r},
    _callGraph{callGraph}
{
  for(clang::CallGraph::iterator node = _callGraph.begin();
      node != _callGraph.end(); ++node)
  {
    if(node->getFirst())
    {
      for(auto child : *(node->getSecond()))
      {
        if(child->getDecl())
          _callers[child->getDecl()].push_back(node->getFirst());
      }
    }
  }
}

void
CompilationTargetAnnotator::addAnnotations()
{
  for(auto decl : _callers)
  {
    bool isHost = false;
    bool isDevice = false;

    if(decl.first)
      correctFunctionAnnotations(isHost, isDevice, decl.first);
  }


  for(const Decl* f : _isFunctionCorrectedDevice)
  {
    if(f->getAsFunction())
    {
      HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as __device__: "
                         << f->getAsFunction()->getQualifiedNameAsString() << std::endl;
    }
    writeAnnotation(f, " __device__ ");
  }

  for(const Decl* f: _isFunctionCorrectedHost)
  {
    // Explicit __host__ annotation is only necessary if a __device__ annotation
    // is present as well
    if(isDeviceFunction(f))
    {
      if(f->getAsFunction())
      {
        HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as __host__: "
                           << f->getAsFunction()->getQualifiedNameAsString() << std::endl;
      }
      writeAnnotation(f, " __host__ ");
    }
  }
}

bool
CompilationTargetAnnotator::isHostFunction(const clang::Decl* f) const
{
  return this->containsTargetAttribute(f, "host") ||
      (_isFunctionCorrectedHost.find(f) != _isFunctionCorrectedHost.end());
}

bool
CompilationTargetAnnotator::isDeviceFunction(const clang::Decl* f) const
{
  return this->containsTargetAttribute(f, "device") ||
      (_isFunctionCorrectedDevice.find(f) != _isFunctionCorrectedDevice.end());
}

bool
CompilationTargetAnnotator::isKernelFunction(const clang::Decl* f) const
{
  return this->containsTargetAttribute(f, "kernel");
}

bool
CompilationTargetAnnotator::containsTargetAttribute(
    const Decl* f,
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


bool CompilationTargetAnnotator::canCallHostFunctions(
    const clang::Decl* f) const
{
  return isHostFunction(f);
}

bool CompilationTargetAnnotator::canCallDeviceFunctions(
    const clang::Decl* f) const
{
  return isDeviceFunction(f) || isKernelFunction(f);
}


/// Determines if a function is called by host or device functions.
void
CompilationTargetAnnotator::correctFunctionAnnotations(
    bool& isHost, bool& isDevice, const clang::Decl* f)
{
  if(!f)
    return;

  if(isKernelFunction(f))
  {
    // Treat kernels as device functions since they execute on the device
    isDevice = true;
    isHost = false;
  }
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

      isHost = isHostFunction(f);
      isDevice = isDeviceFunction(f);

      auto callers = _callers[f];
#ifdef HIPSYCL_VERBOSE_DEBUG
      if(f->getAsFunction())
      {
        HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: "
                           << "call graph for "
                           << f->getAsFunction()->getQualifiedNameAsString() << ": " << std::endl;
      }
#endif
      for(const clang::Decl* caller : callers)
      {
#ifdef HIPSYCL_VERBOSE_DEBUG
        if(caller->getAsFunction())
        {
          HIPSYCL_DEBUG_INFO << "    called by "
                             << caller->getAsFunction()->getQualifiedNameAsString() << std::endl;
        }
#endif
        if(isHost && isDevice)
          break;

        bool callerIsHostFunction = false;
        bool callerIsDeviceFunction = false;

        correctFunctionAnnotations(callerIsHostFunction, callerIsDeviceFunction, caller);

        isHost |= callerIsHostFunction;
        isDevice |= callerIsDeviceFunction;  
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

void CompilationTargetAnnotator::writeAnnotation(
    const clang::Decl* f,
    const std::string& annotation)
{
  for(const Decl* currentDecl = f->getMostRecentDecl();
      currentDecl != nullptr;
      currentDecl = currentDecl->getPreviousDecl())
  {
    if(isa<FunctionDecl>(currentDecl))
    {
      const clang::FunctionDecl* f = cast<const clang::FunctionDecl>(currentDecl);
      _rewriter.InsertText(f->getTypeSpecStartLoc(), annotation);
    }
  }
}

void CompilationTargetAnnotator::markAsHost(const clang::Decl* f)
{
  if(isKernelFunction(f))
    return;

  if(!isHostFunction(f))
    _isFunctionCorrectedHost.insert(f);

}

void CompilationTargetAnnotator::markAsDevice(const clang::Decl* f)
{
  if(isKernelFunction(f))
    return;

  if(!isDeviceFunction(f))
    _isFunctionCorrectedDevice.insert(f);
}


}
}

