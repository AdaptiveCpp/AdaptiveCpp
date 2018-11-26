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
#include "Attributes.hpp"

//#define HIPSYCL_VERBOSE_DEBUG

using namespace clang;

namespace hipsycl {
namespace transform {

CompilationTargetAnnotator::CompilationTargetAnnotator(Rewriter& r,
                                                       CallGraph& callGraph)
  : _rewriter{r},
    _callGraph{callGraph}
{
  for(CallGraph::iterator node = _callGraph.begin();
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
    else
    {
      for(auto child : *(node->getSecond()))
        if(child->getDecl())
          if(_callers.find(child->getDecl()) == _callers.end())
            _callers[child->getDecl()] = std::vector<const clang::Decl*>();
    }
  }
}

void
CompilationTargetAnnotator::treatConstructsAsFunctionCalls(
    const CXXConstructCallerMatcher& constructCallers)
{
  for(const auto& construct : constructCallers.getResults())
  {
    for(const clang::Decl* caller : construct.second)
    {
      _callers[construct.first].push_back(caller);

      // Also add destructor as a callee
      if(isa<CXXConstructorDecl>(construct.first))
      {
        const CXXRecordDecl* parent =
            cast<CXXConstructorDecl>(construct.first)->getParent();
        if(parent)
        {
          const CXXDestructorDecl* destructor = parent->getDestructor();

          if(destructor)
            _callers[destructor].push_back(caller);
        }

      }
    }
  }
}

void
CompilationTargetAnnotator::addAnnotations()
{
  // Build _callees
  _callees.clear();
  _isFunctionProcessed.clear();

  for(auto decl : _callers)
  {
    for(auto caller : decl.second)
      _callees[caller].push_back(decl.first);
  }

  // We do two sweeps: First, we correct the __host__/__device__
  // attributes of all functions based on the functions that call a function;
  // Afterwards we do a second sweep and update the attribute
  // based on the functions called by a function.
  // The second sweep also allows us to correctly treat device functions
  // that are not actually called by a kernel (e.g. unused functions in libraries)
  // but that make calls to __device__ functions (e.g. math functions, intrinsics etc)
  for(auto decl : _callers)
  {
    bool isHost = false;
    bool isDevice = false;

    if(decl.first)
    {
      // This corrects device, host attributes
      correctFunctionAnnotations(isHost, isDevice, decl.first,
                                 targetDeductionDirection::fromCaller);

      // If this is a kernel function in a parallel hierarchical for,
      // we also need to add __shared__ attributes. This is the case
      // if the function is called by cl::sycl::detail::dispatch::parallel_for_workgroup
      for(auto caller : decl.second)
      {
        if(caller && caller->getAsFunction())
          if(caller->getAsFunction()->getQualifiedNameAsString() ==
             "cl::sycl::detail::dispatch::parallel_for_workgroup")
            this->correctSharedMemoryAnnotations(decl.first);
      }
    }
  }

  // Forget which functions we have processed since we now start over
  // with the second sweep
  this->_isFunctionProcessed.clear();
  // Second sweep - update host/device attributes based on called functions
#ifdef HIPSYCL_TRANSFORM_INVESTIGATE_CALLEES
  for(auto decl : _callees)
  {
    bool isHost = false;
    bool isDevice = false;
    if(decl.first)
    {
      correctFunctionAnnotations(isHost, isDevice, decl.first,
                                 targetDeductionDirection::fromCallee);
    }
  }
#endif

  // Write out attributes
  for(const Decl* f : _isFunctionCorrectedDevice)
  {
    writeAnnotation(f, " __device__ ");
  }

  for(const Decl* f: _isFunctionCorrectedHost)
  {
    // Explicit __host__ annotation is only necessary if a __device__ annotation
    // is present as well
    if(isDeviceFunction(f))
    {
      writeAnnotation(f, " __host__ ");
    }
  }
}

bool
CompilationTargetAnnotator::isPrivateMemory(const clang::DeclStmt* declaration) const
{
  for(auto decl = declaration->decl_begin();
      decl != declaration->decl_end();
      ++decl)
  {
    if(clang::isa<clang::VarDecl>(*decl))
    {
      const clang::VarDecl* var = clang::cast<clang::VarDecl>(*decl);
      const clang::CXXRecordDecl* recordDecl = var->getType()->getAsCXXRecordDecl();
      if(recordDecl)
        return recordDecl->getQualifiedNameAsString() == "cl::sycl::private_memory";
    }
  }

  return false;
}

void
CompilationTargetAnnotator::correctSharedMemoryAnnotations(
    const clang::Decl* kernelFunction)
{
  if(const clang::Stmt* body = kernelFunction->getBody())
  {
    for(auto currentStmt = body->child_begin();
        currentStmt != body->child_end(); ++currentStmt)
    {
      if(clang::isa<clang::DeclStmt>(*currentStmt))
      {
        if(!isPrivateMemory(clang::cast<clang::DeclStmt>(*currentStmt)))
        {
          HIPSYCL_DEBUG_INFO << "Marking variable as __shared__ in "
                             << kernelFunction->getAsFunction()->getQualifiedNameAsString()
                             << std::endl;
#if LLVM_VERSION_MAJOR > 6
          _rewriter.InsertText((*currentStmt)->getBeginLoc(), " __shared__ ");
#else
          _rewriter.InsertText((*currentStmt)->getLocStart(), " __shared__ ");
#endif
        }
      }
    }
  }
}

bool
CompilationTargetAnnotator::isHostFunction(const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<HostAttribute>(f) ||
      (_isFunctionCorrectedHost.find(f) != _isFunctionCorrectedHost.end());
}

bool
CompilationTargetAnnotator::isDeviceFunction(const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<DeviceAttribute>(f) ||
      (_isFunctionCorrectedDevice.find(f) != _isFunctionCorrectedDevice.end());
}

bool
CompilationTargetAnnotator::isKernelFunction(const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<KernelAttribute>(f);
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
    bool& isHost, bool& isDevice, const clang::Decl* f,
    CompilationTargetAnnotator::targetDeductionDirection direction)
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

      // If the deduction direction is fromCaller, we investigate
      // the functions that call this function and infer the host/device
      // attributes from that
      // [rationale: A function called by a device (host) function must also
      //  be compiled for device (host)].
      // Otherwise (if deduction direction is fromCallee), we investigate
      // the functions called by this function.
      // [rationale: A function calling device (host) functions must also
      // be compiled for device (host)
      const auto& investigatedFunctions =
          (direction == targetDeductionDirection::fromCaller) ?
            _callers[f] : _callees[f];

#ifdef HIPSYCL_VERBOSE_DEBUG
      if(f->getAsFunction())
      {
        HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: "
                           << "call graph for "
                           << f->getAsFunction()->getQualifiedNameAsString() << ": " << std::endl;
      }
#endif

      for(const clang::Decl* callerOrCallee : investigatedFunctions)
      {
#ifdef HIPSYCL_VERBOSE_DEBUG
        if(callerOrCallee->getAsFunction())
        {
          auto functionName = callerOrCallee->getAsFunction()->getQualifiedNameAsString();

          if(direction == targetDeductionDirection::fromCaller)
          {
            HIPSYCL_DEBUG_INFO << "    called by "
                               << functionName
                               << std::endl;
          }
          else
          {
            HIPSYCL_DEBUG_INFO << "    calling "
                               << functionName
                               << std::endl;
          }
        }
#endif
        // This does not really save us any work, we must go through
        // the entire call graph anyway, and we won't visit the same
        // node twice because of the _isFunctionProcessed map.
        //if(isHost && isDevice)
        //  break;

        bool investigatedFunctionIsHostFunction = false;
        bool investigatedFunctionIsDeviceFunction = false;

        correctFunctionAnnotations(investigatedFunctionIsHostFunction,
                                   investigatedFunctionIsDeviceFunction,
                                   callerOrCallee,
                                   direction);

        isHost |= investigatedFunctionIsHostFunction;
        isDevice |= investigatedFunctionIsDeviceFunction;

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
      bool suppressAnnotation = false;


      if(isa<CXXMethodDecl>(currentDecl))
      {
        // Don't add __host__,__device__ attributes to defaulted or deleted
        // functions
        const CXXMethodDecl* methodDecl = cast<CXXMethodDecl>(currentDecl);
        if(methodDecl->isDefaulted() || methodDecl->isDeleted())
          suppressAnnotation = true;
      }

      if(!suppressAnnotation)
      {
        if(isa<CXXConstructorDecl>(currentDecl) || isa<CXXDestructorDecl>(currentDecl))
        {
          _rewriter.InsertText(f->getLocStart(), annotation);
        }
        else
        {
          const clang::FunctionDecl* f = cast<const clang::FunctionDecl>(currentDecl);
          _rewriter.InsertText(f->getTypeSpecStartLoc(), annotation);
        }

        if(f->getAsFunction())
        {
          HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as '"
                             << annotation << "': "
                             << f->getAsFunction()->getQualifiedNameAsString() << std::endl;
        }
      }
    }
    else
    {
      _rewriter.InsertText(f->getLocStart(), annotation);
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

