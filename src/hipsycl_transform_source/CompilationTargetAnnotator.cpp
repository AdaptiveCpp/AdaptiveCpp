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

  // We do two sweeps: The first sweep allows us to correctly treat device functions
  // that are not actually called by a kernel (e.g. unused functions in libraries)
  // but that make calls to __device__ functions (e.g. math functions, intrinsics etc)
  // In a second sweep, we correct the __host__/__device__
  // attributes of all functions based on the functions that call a function.


  // First sweep - update host/device attributes based on called pure
  // __device__ functions
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
  // Forget which functions we have processed since we now start over
  // with the second sweep
  this->_isFunctionProcessed.clear();

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


  // Write out attributes
  for(auto f : _isFunctionCorrectedDevice)
  {
    writeAnnotation(f.second, " __device__ ");
  }

  for(auto f: _isFunctionCorrectedHost)
  {
    // Explicit __host__ annotation is only necessary if a __device__ annotation
    // is present as well
    if(isDeviceFunction(f.second))
    {
      writeAnnotation(f.second, " __host__ ");
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
  auto declaration_key = this->getDeclKey(f);
  return this->containsAttributeForCompilation<HostAttribute>(f) ||
      (_isFunctionCorrectedHost.find(declaration_key) != _isFunctionCorrectedHost.end());
}

bool
CompilationTargetAnnotator::isDeviceFunction(const clang::Decl* f) const
{
  auto declaration_key = this->getDeclKey(f);
  return this->containsAttributeForCompilation<DeviceAttribute>(f) ||
      (_isFunctionCorrectedDevice.find(declaration_key) != _isFunctionCorrectedDevice.end());
}

bool
CompilationTargetAnnotator::isKernelFunction(const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<KernelAttribute>(f);
}

CompilationTargetAnnotator::FunctionLocationIdentifier
CompilationTargetAnnotator::getDeclKey(const clang::Decl* f) const
{
  // In order to avoid having different Decl's as keys for different
  // instantiations of the same template, we use the position
  // in the source code of the body as key. This position
  // is of course invariant under different template instantiations,
  // which allows us treat Decl's in the call graph that are
  // logically different (because they are different instantiations)
  // as one actual Decl, as in the source.
  // ToDo: Surely clang must have a better way to find the "general"
  // Decl of template instantiations?
  if(f->hasBody())
    return f->getBody()->getLocStart().printToString(_rewriter.getSourceMgr());
  else
    return f->getLocStart().printToString(_rewriter.getSourceMgr());
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
    if(direction == targetDeductionDirection::fromCaller)
    {
      // Treat kernels as device functions since they execute on the device
      // and all functions called by kernels are device functions
      isDevice = true;
      isHost = false;
    }
    else
    {
      // If we are investigating callees, treat kernels as host functions
      // since they are called on the host
      isDevice = false;
      isHost = true;
    }
  }
  else
  {
    isHost = isHostFunction(f);
    isDevice = isDeviceFunction(f);

    if(!_isFunctionProcessed[f])
    {
      // Set this already in the beginning to avoid getting stuck in
      // cycles in the call graph
      _isFunctionProcessed[f] = true;

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

      for(const clang::Decl* callerOrCallee : investigatedFunctions)
      {
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


        if(direction == targetDeductionDirection::fromCaller)
        {
          // If we are not looking for calls to __device__ functions,
          // we need to become a __host__ function if we are called from
          // other __host__ functions.
          // But we don't necessarily need to become a __host__
          // function if we're calling a __host__ __device__ function.
          // So, the __host__ attribute should only be taken into
          // account if we are deducing attributes based on our callers.

          isHost = isHost || investigatedFunctionIsHostFunction;

          // If we are calling a kernel function and are trying
          // to deduce our attributes based on the functions we call,
          // we must ignore the returned results since we would otherwise
          // wrongly conclude that a function calling a kernel must be
          // compiled for __device__.
          // But, if we're inferring the attributes from our callers,
          // we must take kernels into account since a function called
          // from a kernel definitely must be compiled for device.
          // We therefore need two different code paths depending on the
          // direction of deduction.

          isDevice = isDevice || investigatedFunctionIsDeviceFunction;

#ifdef HIPSYCL_VERBOSE_DEBUG
          HIPSYCL_DEBUG_INFO << "Execution space deduction of "
                             << f->getAsFunction()->getQualifiedNameAsString()
                             << "... " << std::endl;
          if (investigatedFunctionIsHostFunction)
            HIPSYCL_DEBUG_INFO
                      << " ... is __host__ because: Called by __host__ function "
                      << callerOrCallee->getAsFunction()->getQualifiedNameAsString()
                      << std::endl;
          if (investigatedFunctionIsDeviceFunction)
            HIPSYCL_DEBUG_INFO
                      << " ... is __device__ because: Called by __device__ function "
                      << callerOrCallee->getAsFunction()->getQualifiedNameAsString()
                      << std::endl;
#endif
        }
        else
        {
          // At the moment, we only mark callers as __device__ if the
          // callees are 100% device functions. If they are also __host__,
          // it is better to assume that the caller is is a __host__ function
          // to avoid spamming everything (up to main()) with __device__ attributes
          // as soon as one __host__ __device__ function is called.

          if(!isHostFunction(callerOrCallee)
             && !isKernelFunction(callerOrCallee))
          {
            isDevice |= investigatedFunctionIsDeviceFunction;

#ifdef HIPSYCL_VERBOSE_DEBUG
            HIPSYCL_DEBUG_INFO << "Execution space deduction of "
                               << f->getAsFunction()->getQualifiedNameAsString()
                               << "... " << std::endl;
            if (investigatedFunctionIsDeviceFunction)
            {
              HIPSYCL_DEBUG_INFO
                        << " ... is __device__ because: Calls __device__ function "
                        << callerOrCallee->getAsFunction()->getQualifiedNameAsString()
                        << std::endl;
            }
#endif
          }
        }

      }

      // If we deduce from the callees, we can only deduce __device__
      // attributes
      if(direction == targetDeductionDirection::fromCaller)
      {
        if(isHost)
          markAsHost(f);
      }
      if(isDevice)
        markAsDevice(f);
    }
    // If device isn't explicitly marked as __host__ or __device__, treat it as
    // host
    if(!isDevice && !isHost && direction == targetDeductionDirection::fromCaller)
    {
#ifdef HIPSYCL_VERBOSE_DEBUG
      HIPSYCL_DEBUG_INFO << "Execution space deduction of "
                         << f->getAsFunction()->getQualifiedNameAsString()
                         << "... " << std::endl;
      HIPSYCL_DEBUG_INFO << " ... is __host__ because: No contrary evidence found."
                         << std::endl;
#endif
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

  auto id = this->getDeclKey(f);
  if(!isHostFunction(f))
    _isFunctionCorrectedHost[id] = f;
}

void CompilationTargetAnnotator::markAsDevice(const clang::Decl* f)
{
  if(isKernelFunction(f))
    return;

  auto id = this->getDeclKey(f);
  if(!isDeviceFunction(f))
    _isFunctionCorrectedDevice[id] = f;
}


}
}

