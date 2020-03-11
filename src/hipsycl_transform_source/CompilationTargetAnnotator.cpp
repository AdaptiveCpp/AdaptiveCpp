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

#include "hipSYCL/sycl/detail/debug.hpp"

#include "CompilationTargetAnnotator.hpp"
#include "Attributes.hpp"

//#define HIPSYCL_VERBOSE_DEBUG
//#define HIPSYCL_DUMP_CALLGRAPH

using namespace clang;

namespace hipsycl {
namespace transform {

template<class T>
clang::SourceLocation getBegin(const T* ast_element)
{
#if LLVM_VERSION_MAJOR > 6
  return ast_element->getBeginLoc();
#else
  return ast_element->getLocStart();
#endif
}


CompilationTargetAnnotator::CompilationTargetAnnotator(Rewriter& r,
                                                       CallGraph& callGraph)
  : _rewriter{r},
    _callGraph{callGraph}
{
  this->_warningAttemptedExternalModificationID = 
    _rewriter.getSourceMgr().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Attempted to mark function %0 in non-inlined source file as __device__,"
        " is the file in a path where device code is allowed?");


  // When dealing with multiple template instantiations, we may end up
  // with several Decl's (and hence several nodes in the call graph) even
  // if these functions refer to the same location in the source code.
  // Since we are interested in transforming source code, we combine
  // all graph nodes referring to the same source code in one.

  // Go through all nodes and collect synonymous decls
  for(CallGraph::iterator node = _callGraph.begin();
      node != _callGraph.end(); ++node)
  {
    _synonymousDecls[getDeclKey(node->getFirst())].push_back(node->getFirst());
    _containsUnresolvedCalls[node->getFirst()] = 
        node->getSecond()->containsUnresolvedCallExpr();
  }

  // Now we actually start processing the graph - build
  // the _callers map that maps _callers[i] to a vector of caller decls of
  // Decl i.
  for(CallGraph::iterator node = _callGraph.begin();
      node != _callGraph.end(); ++node)
  {
    auto nodeDecl = this->getMainDecl(node->getFirst());
    if(node->getFirst())
    {
      for(auto child : *(node->getSecond()))
      {
        if(child->getDecl())
          _callers[this->getMainDecl(child->getDecl())].push_back(nodeDecl);
      }
    }
    else
    {
      // For decls that have no caller (node->getFirst() == nullptr)
      // we end up in this branch.
      // It is still important to consider these functions! There may
      // be uncalled functions that call __device__ functions and hence
      // must be __device__.
      for(auto child : *(node->getSecond()))
      {
        auto childDecl = getMainDecl(child->getDecl());
        if(_callers.find(childDecl) == _callers.end())
          _callers[childDecl] = std::vector<const clang::Decl*>();
      }
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

          if(destructor && !destructor->isDefaulted())
            _callers[destructor].push_back(caller);
        }

      }
    }
  }
}

void
CompilationTargetAnnotator::addAnnotations()
{
  // Invert _callers map to build _callees
  _callees.clear();
  _isFunctionProcessed.clear();

  for(auto decl : _callers)
    for(auto caller : decl.second)
      _callees[caller].push_back(decl.first);

#if defined(HIPSYCL_VERBOSE_DEBUG) && defined(HIPSYCL_DUMP_CALLGRAPH)
  for(auto decl : _callees)
  {
    if (decl.first)
    {
      HIPSYCL_DEBUG_INFO
          << decl.first->getAsFunction()->getQualifiedNameAsString() << " ["
          << decl.first << "] calls: " << std::endl;
    }
    for(auto callee : decl.second)
    {
      if (callee) 
      {
        HIPSYCL_DEBUG_INFO
            << " ---> " << callee->getAsFunction()->getQualifiedNameAsString()
            << std::endl;
      }
    }
  }
#endif

  // First correct __shared__ attributes and mark
  // all kernels as __device__
  for(auto decl : _callers)
  {
    if(decl.first)
    {
      for(auto caller : decl.second)
      {
        if(caller && caller->getAsFunction())
        {
          std::string callerName = caller->getAsFunction()->getQualifiedNameAsString();
          if(callerName.find("hipsycl::sycl::detail::dispatch::device::") !=
             std::string::npos)
          {
            // If this is a kernel function in a parallel hierarchical for,
            // we need to add __shared__ attributes. This is the case
            // if the function is called by hipsycl::sycl::detail::dispatch::device::parallel_for_workgroup
            if(callerName == "hipsycl::sycl::detail::dispatch::device::parallel_for_workgroup")
              this->correctSharedMemoryAnnotations(decl.first);

            // In any way, mark all kernels as __device__ to
            // make our life easier later on
            this->markAsDevice(decl.first);
          }

        }
      }
    }
  }

  // Now worry about __host__ and __device__ attributes of
  // the remaining call graph:
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

void 
CompilationTargetAnnotator::pruneUninstantiatedTemplates()
{
  // Prune uninstantiated templates
  for(auto synonymousDeclGroup : _synonymousDecls)
  {
    bool allDeclsContainUnresolvedCalls = true;

    for(auto decl : synonymousDeclGroup.second)
    {
      if (!this->_containsUnresolvedCalls[decl]) 
      {
        allDeclsContainUnresolvedCalls = false;
        break;
      }
    }

    if(allDeclsContainUnresolvedCalls)
    {
      const clang::Decl* d = synonymousDeclGroup.second.front();
      HIPSYCL_DEBUG_INFO << "Removing function body of " 
                         << d->getAsFunction()->getQualifiedNameAsString()
                         << " (all declarations contain unresolved function calls)"
                         << std::endl;

      if (d->hasBody()) {

        const clang::Stmt *body = d->getBody();
        auto bodyStart = body->getSourceRange().getBegin();

        auto bodyEnd = body->getSourceRange().getEnd();
      
        _rewriter.InsertTextAfterToken(bodyStart,
            "\n#if 0 // -- definition stripped by hipsycl_transform_source\n");
        _rewriter.InsertText(bodyEnd, "\n#endif\n");
        // TODO: Correct the current #line in the source
      }
    }
  } 
}

const clang::Decl*
CompilationTargetAnnotator::getMainDecl(const clang::Decl* decl) const
{
  auto key = this->getDeclKey(decl);
  return this->_synonymousDecls.at(key).front();
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
        return recordDecl->getQualifiedNameAsString() == "hipsycl::sycl::private_memory";
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

          _rewriter.InsertText(getBegin(*currentStmt), " __shared__ ");
        }
      }
    }
  }
}

bool
CompilationTargetAnnotator::isExplicitlyHostFunction(
    const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<HostAttribute>(f);
}

bool
CompilationTargetAnnotator::isExplicitlyDeviceFunction(
    const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<DeviceAttribute>(f);
}

bool
CompilationTargetAnnotator::isHostFunction(const clang::Decl* f) const
{
  auto declaration_key = this->getDeclKey(f);
  return isExplicitlyHostFunction(f) ||
      (_isFunctionCorrectedHost.find(declaration_key) != _isFunctionCorrectedHost.end());
}

bool
CompilationTargetAnnotator::isDeviceFunction(const clang::Decl* f) const
{
  auto declaration_key = this->getDeclKey(f);
  return isExplicitlyDeviceFunction(f) ||
      (_isFunctionCorrectedDevice.find(declaration_key) != _isFunctionCorrectedDevice.end());
}

bool
CompilationTargetAnnotator::isKernelFunction(const clang::Decl* f) const
{
  return this->containsAttributeForCompilation<KernelAttribute>(f);
}

CompilationTargetAnnotator::DeclIdentifier
CompilationTargetAnnotator::getDeclKey(const clang::Decl* f) const
{
  // In order to avoid having different Decl's as keys for different
  // instantiations of the same template, we use the position
  // in the source code of the body as key. This position
  // is of course invariant under different template instantiations,
  // which allows us treat Decl's in the call graph that are
  // logically different (because they are different instantiations)
  // as one actual Decl, as in the source.
  // TODO: Surely clang must have a better way to find the "general"
  // Decl of template instantiations?
  if (!f)
    return "<nullptr>";
  if(f->hasBody())
    return getBegin(f->getBody()).printToString(_rewriter.getSourceMgr());
  else
    return getBegin(f).printToString(_rewriter.getSourceMgr());
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
    // If the function has been marked explicitly as host
    // or device, don't modify it further
    isHost = isExplicitlyHostFunction(f);
    isDevice = isExplicitlyDeviceFunction(f);
    if(isHost || isDevice)
    {
      _isFunctionProcessed[f] = true;
      return;
    }

    // Otherwise, we also retrieve the current information
    // about the deduction state
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
          if (investigatedFunctionIsHostFunction) {
            HIPSYCL_DEBUG_INFO
                << " ... is __host__ because: Called by __host__ function "
                << callerOrCallee->getAsFunction()->getQualifiedNameAsString()
                << std::endl;
          }
          if (investigatedFunctionIsDeviceFunction) {
            HIPSYCL_DEBUG_INFO
                << " ... is __device__ because: Called by __device__ function "
                << callerOrCallee->getAsFunction()->getQualifiedNameAsString()
                << std::endl;
          }
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
    // If function isn't explicitly marked as __host__ or __device__, treat it as
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
        clang::SourceLocation insertLoc;

        if(isa<CXXConstructorDecl>(currentDecl) || isa<CXXDestructorDecl>(currentDecl))
        {
          insertLoc = getBegin(f);
        }
        else
        {
          const clang::FunctionDecl* f = cast<const clang::FunctionDecl>(currentDecl);
          insertLoc = f->getTypeSpecStartLoc();
        }

        std::string functionName;

        if(f->getAsFunction())
        {
          functionName = f->getAsFunction()->getQualifiedNameAsString();
          HIPSYCL_DEBUG_INFO << "hipsycl_transform_source: Marking function as '"
                             << annotation << "': "
                             << functionName << std::endl;
        }

#ifdef ENABLE_WARNING_IF_EXTERNAL_MODIFICATION_ATTEMPTED
        if(annotation != " __host__ " && 
          (_rewriter.getSourceMgr().getMainFileID() !=
           _rewriter.getSourceMgr().getFileID(insertLoc)))
        {
          _rewriter.getSourceMgr().getDiagnostics().Report(insertLoc, 
                this->_warningAttemptedExternalModificationID).AddString(functionName);
        }
#endif
        _rewriter.InsertText(insertLoc, annotation);  
      }
    }
    else
    {
      _rewriter.InsertText(getBegin(f), annotation);
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

