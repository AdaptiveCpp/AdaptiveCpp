/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_FRONTEND_HPP
#define HIPSYCL_FRONTEND_HPP

#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <regex>
#include <sstream>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Sema/Sema.h"

#include "CompilationState.hpp"
#include "Attributes.hpp"

#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace compiler {
namespace detail {

///
/// Utility type to generate the set of all function declarations
/// implictly or explictly reachable from some initial declaration.
///
class CompleteCallSet : public clang::RecursiveASTVisitor<CompleteCallSet> {
  public:
    using FunctionSet = std::unordered_set<clang::FunctionDecl*>;

    CompleteCallSet(clang::Decl* D)
    {
      TraverseDecl(D);
    }

    bool VisitFunctionDecl(clang::FunctionDecl* FD)
    {
      visitedDecls.insert(FD);
      return true;
    }

    bool VisitCallExpr(clang::CallExpr* CE)
    {
      if(auto Callee = CE->getDirectCallee())
        TraverseDecl(Callee);
      return true;
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr* CE)
    {
      if(auto Callee = CE->getConstructor())
      {
        TraverseDecl(Callee);
        // Since for destructor calls no explicit AST nodes are created, we simply use this opportunity to
        // find the corresponding destructor for all constructed types (since we assume that every
        // type that can be constructed on the GPU also can and will be destructed).
        if(auto Ptr = llvm::dyn_cast_or_null<clang::PointerType>(Callee->getThisType()->getCanonicalTypeUnqualified()))
          if(auto Record = llvm::dyn_cast<clang::RecordType>(Ptr->getPointeeType()))
            if(auto RecordDecl = llvm::dyn_cast<clang::CXXRecordDecl>(Record->getDecl()))
              if(auto DtorDecl = RecordDecl->getDestructor())
                TraverseDecl(DtorDecl);
      }
      return true;
    }

    bool TraverseDecl(clang::Decl* D)
    {
      clang::Decl* DefinitionDecl = D;
      clang::FunctionDecl* FD = clang::dyn_cast<clang::FunctionDecl>(D);

      if(FD){
        const clang::FunctionDecl* ActualDefinition;
        if(FD->isDefined(ActualDefinition)) {
          
          DefinitionDecl = const_cast<clang::FunctionDecl*>(ActualDefinition);
        }
      }

      if (visitedDecls.find(llvm::dyn_cast_or_null<clang::FunctionDecl>(
              DefinitionDecl)) == visitedDecls.end())
        return clang::RecursiveASTVisitor<CompleteCallSet>::TraverseDecl(
            DefinitionDecl);
      
      return true;
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool shouldVisitImplicitCode() const { return true; }

    const FunctionSet& getReachableDecls() const { return visitedDecls; }

  private:
    FunctionSet visitedDecls;
};

///
/// Builds a kernel name from a RecordDecl, taking into account template specializations.
/// Returns an empty string if the name is not a valid kernel name.
///
inline std::string buildKernelNameFromRecordType(const clang::QualType &RecordType, clang::MangleContext *Mangler) {
  std::string KernelName;
  llvm::raw_string_ostream SS(KernelName);
  Mangler->mangleTypeName(RecordType, SS);

  return KernelName;
}

inline std::string buildKernelName(clang::TemplateArgument SyclTagTypeTA, clang::MangleContext *Mangler) {
  assert(SyclTagTypeTA.getKind() == clang::TemplateArgument::ArgKind::Type);
  auto SyclTagType = SyclTagTypeTA.getAsType();
  auto RecordType = llvm::dyn_cast<clang::RecordType>(SyclTagType.getTypePtr());
  if(RecordType == nullptr || !llvm::isa<clang::RecordDecl>(RecordType->getDecl()))
  {
    // We only support structs/classes as kernel names
    return "";
  }
  auto DeclName = buildKernelNameFromRecordType(SyclTagType, Mangler);
  return "__hipsycl_kernel_" + DeclName;
}

}

class FrontendASTVisitor : public clang::RecursiveASTVisitor<FrontendASTVisitor>
{
  clang::CompilerInstance &Instance;
  
public:
  FrontendASTVisitor(clang::CompilerInstance &instance)
      : Instance{instance}
      , KernelNameMangler(instance.getASTContext().createMangleContext())
  {}

  ~FrontendASTVisitor()
  {}
  
  bool shouldVisitTemplateInstantiations() const { return true; }

  /// Return whether this visitor should recurse into implicit
  /// code, e.g., implicit constructors and destructors.
  bool shouldVisitImplicitCode() const { return true; }
  
  // We also need to have look at all statements to identify Lambda declarations
  bool VisitStmt(clang::Stmt *S) {

    if(clang::isa<clang::LambdaExpr>(S))
    {
      clang::LambdaExpr* lambda = clang::cast<clang::LambdaExpr>(S);
      clang::FunctionDecl* callOp = lambda->getCallOperator();
      if(callOp)
        this->VisitFunctionDecl(callOp);
    }
    
    return true;
  }
  
  bool VisitDecl(clang::Decl* D){
    if(clang::VarDecl* V = clang::dyn_cast<clang::VarDecl>(D)){
      if(isLocalMemory(V))
        // Maybe we should additionally check that this is in kernels?
        storeVariableInLocalMemory(V);
    }
    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *f) {
    if(!f)
      return true;

    this->processFunctionDecl(f);

    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    auto F = llvm::dyn_cast_or_null<clang::FunctionDecl>(Call->getDirectCallee());
    if(!F) return true;
    if(!CustomAttributes::SyclKernel.isAttachedTo(F)) return true;

    auto KernelFunctorType = llvm::dyn_cast<clang::RecordType>(Call->getArg(0)
      ->getType()->getCanonicalTypeUnqualified());

    // Store user kernel for it to be marked as device code later on
    if(KernelFunctorType)
    {
      auto Methods = llvm::dyn_cast<clang::CXXRecordDecl>(
        KernelFunctorType->getDecl())->methods();
      for(auto&& M : Methods)
      {
        if(M->getNameAsString() == "operator()")
          UserKernels.insert(M);
      }
    }

    // Determine unique kernel name to be used for symbol name in device IR
    clang::FunctionTemplateSpecializationInfo* Info = F->getTemplateSpecializationInfo();

    // Check whether a unique kernel name is required. If no name is provided and the
    // functor is not a lambda, we allow it and simply do nothing.
    bool NameRequired = true;

    const auto KernelNameArgument = Info->TemplateArguments->get(0);
    if (KernelNameArgument.getKind() == clang::TemplateArgument::ArgKind::Type) {
      if (auto RecordType = llvm::dyn_cast<clang::RecordType>(KernelNameArgument.getAsType().getTypePtr())) {
        const auto RecordDecl = RecordType->getDecl();
        auto KernelName = RecordDecl->getNameAsString();
        if (KernelName == "_unnamed_kernel") {
          // If no name is provided, rely on clang name mangling

          // Earlier clang versions suffer from potentially inconsistent
          // lambda numbering across host and device passes
#if LLVM_VERSION_MAJOR < 10
          const auto KernelFunctorArgument = Info->TemplateArguments->get(1);
          
          if (KernelFunctorArgument.getAsType().getTypePtr()->getAsCXXRecordDecl() &&
               KernelFunctorArgument.getAsType().getTypePtr()->getAsCXXRecordDecl()->isLambda())
          {
            auto SL = llvm::dyn_cast<clang::CXXRecordDecl>(
                KernelFunctorType->getDecl())->getSourceRange().getBegin();
            auto ID = Instance.getASTContext().getDiagnostics()
              .getCustomDiagID(clang::DiagnosticsEngine::Level::Error,
                  "Optional kernel lambda naming requires clang >= 10");
            Instance.getASTContext().getDiagnostics().Report(SL, ID);
          }
#endif

          NameRequired = false;
        }
      }
    }

    if (NameRequired)
    {
      std::string KernelName = detail::buildKernelName(KernelNameArgument, KernelNameMangler.get());

      // Abort with error diagnostic if no kernel name could be built
      if(KernelName.empty())
      {
        // Since we cannot easily get the source location of the template
        // specialization where the name is passed by the user (e.g. a
        // parallel_for call), we attach the diagnostic to the kernel
        // functor instead.
        // TODO: Improve on this.
        auto SL = llvm::dyn_cast<clang::CXXRecordDecl>(
            KernelFunctorType->getDecl())->getSourceRange().getBegin();
        auto ID = Instance.getASTContext().getDiagnostics()
          .getCustomDiagID(clang::DiagnosticsEngine::Level::Error,
              "Not a valid kernel name: %0");
        Instance.getASTContext().getDiagnostics().Report(SL, ID) <<
          Info->TemplateArguments->get(0);
      }

      // Add the AsmLabel attribute which, if present,
      // is used by Clang instead of the function's mangled name.
      F->addAttr(clang::AsmLabelAttr::CreateImplicit(Instance.getASTContext(),
            KernelName));
      HIPSYCL_DEBUG_INFO << "AST processing: Adding ASM label attribute with kernel name "
        << KernelName << "\n";
    }

    return true;
  }

  void applyAttributes()
  {
    for(auto F : MarkedHostDeviceFunctions)
    {
      // Strictly speaking, setting these attributes is not even necessary!
      // It's only important that the kernel has the right attribute.
      if (!F->hasAttr<clang::CUDAHostAttr>())
        F->addAttr(clang::CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
      if (!F->hasAttr<clang::CUDADeviceAttr>())
        F->addAttr(clang::CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));
    }

    for(auto F : MarkedKernels)
    {
      if (!F->hasAttr<clang::CUDAGlobalAttr>() &&
          CustomAttributes::SyclKernel.isAttachedTo(F)) {

        auto* NewAttr = clang::CUDAGlobalAttr::CreateImplicit(Instance.getASTContext());

        F->addAttr(NewAttr);

        // create amdgpu_flat_work_group_size attribute to allow sub_groups outside of [128, 256]
        // first we need to create Expressions containing the workgroup-sizes
        auto sizeType = Instance.getASTContext().getSizeType();
        auto minFlatWorkgroupSize = llvm::APInt(Instance.getASTContext().getTypeSize(sizeType), 64);
        auto maxFlatWorkgroupSize = llvm::APInt(Instance.getASTContext().getTypeSize(sizeType), 1024);
        auto* minFlatWorkgroupExpr = clang::IntegerLiteral::Create(Instance.getASTContext(), minFlatWorkgroupSize, sizeType, clang::SourceLocation{});
        auto* maxFlatWorkgroupExpr = clang::IntegerLiteral::Create(Instance.getASTContext(), maxFlatWorkgroupSize, sizeType, clang::SourceLocation{});

        // to finally create the attribute itself
        auto * NewAMDGPUAttr = clang::AMDGPUFlatWorkGroupSizeAttr::CreateImplicit(Instance.getASTContext(), minFlatWorkgroupExpr , maxFlatWorkgroupExpr);

        F->addAttr(NewAMDGPUAttr);
      }
    }

    for(auto F : UserKernels)
    {
      std::unordered_set<clang::FunctionDecl*> UserKernels;

      // Mark all functions called by user kernels as host / device.
      detail::CompleteCallSet CCS(F);
      for (auto&& RD : CCS.getReachableDecls())
      {
        HIPSYCL_DEBUG_INFO << "AST processing: Marking function as __host__ __device__: "
                           << RD->getQualifiedNameAsString() << std::endl;
        markAsHostDevice(RD);
        if (!RD->hasAttr<clang::CUDAHostAttr>())
          RD->addAttr(clang::CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
        if (!RD->hasAttr<clang::CUDADeviceAttr>())
          RD->addAttr(clang::CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));
      }
    }
  }

  std::unordered_set<clang::FunctionDecl*>& getMarkedHostDeviceFunctions()
  {
    return MarkedHostDeviceFunctions;
  }

  std::unordered_set<clang::FunctionDecl*>& getKernels()
  {
    return MarkedKernels;
  }

private:
  std::unordered_set<clang::FunctionDecl*> MarkedHostDeviceFunctions;
  std::unordered_set<clang::FunctionDecl*> MarkedKernels;
  std::unordered_set<clang::FunctionDecl*> UserKernels;
  std::unique_ptr<clang::MangleContext> KernelNameMangler;

  void markAsHostDevice(clang::FunctionDecl* F)
  {
    this->MarkedHostDeviceFunctions.insert(F);
  }

  void markAsKernel(clang::FunctionDecl* F)
  {
    this->MarkedKernels.insert(F);
  }

  void processFunctionDecl(clang::FunctionDecl* f)
  {
    if(!f)
      return;

    if(f->getQualifiedNameAsString() 
        == "hipsycl::glue::hiplike_dispatch::parallel_for_workgroup")
    {
      clang::FunctionDecl* Kernel = f;
      
      HIPSYCL_DEBUG_INFO << "AST Processing: Detected parallel_for_workgroup kernel "
                        << Kernel->getQualifiedNameAsString() << std::endl;

      // Mark local variables as shared memory, unless they are explicitly marked private.
      // Do this not only for the kernel itself, but consider all functions called by the kernel.
      detail::CompleteCallSet CCS(Kernel);
      for(auto&& RD : CCS.getReachableDecls())
      {
        // To prevent every local variable in any function being marked as shared,
        // we only consider functions that receive a hipsycl::sycl::group as their parameter.
        for(auto Param = RD->param_begin(); Param != RD->param_end(); ++Param)
        {
          auto Type = (*Param)->getOriginalType().getTypePtr();
          if(auto DeclType = Type->getAsCXXRecordDecl()) {
            if(DeclType->getQualifiedNameAsString() == "hipsycl::sycl::group")
            {
              storeLocalVariablesInLocalMemory(RD->getBody(), RD);
              break;
            }
          }
        }
      }
    
    }
  
    
    if(CustomAttributes::SyclKernel.isAttachedTo(f)){
      markAsKernel(f); 
    }
  }


  bool isPrivateMemory(const clang::VarDecl* V) const
  {
    const clang::CXXRecordDecl* R = V->getType()->getAsCXXRecordDecl();
    if(R)
      return R->getQualifiedNameAsString() == "hipsycl::sycl::private_memory";
  
    return false;
  }

  bool isLocalMemory(const clang::VarDecl* V) const
  {
    const clang::CXXRecordDecl* R = V->getType()->getAsCXXRecordDecl();
    if(R)
      return R->getQualifiedNameAsString() == "hipsycl::sycl::local_memory";
  
    return false;
  }

  ///
  /// Marks all variable declarations within a given block statement as shared memory,
  /// unless they are explicitly declared as a private memory type.
  ///
  /// Recurses into compound statements (i.e., a set of braces {}).
  ///
  /// NOTE TODO: It is unclear how certain other statement types should be handled.
  /// For example, should the loop variable of a for-loop be marked as shared? Probably not.
  ///
  void storeLocalVariablesInLocalMemory(clang::Stmt* BlockStmt, clang::FunctionDecl* F) const
  {
    for(auto S = BlockStmt->child_begin(); S != BlockStmt->child_end(); ++S)
    {
      if(auto D = clang::dyn_cast<clang::DeclStmt>(*S))
      {
        for(auto decl = D->decl_begin(); decl != D->decl_end(); ++decl)
        {
          if(clang::VarDecl* V = clang::dyn_cast<clang::VarDecl>(*decl))
          {
            if(!isPrivateMemory(V))
              storeVariableInLocalMemory(V);
          }
        }
      }
      else if(clang::dyn_cast<clang::CompoundStmt>(*S))
      {
        storeLocalVariablesInLocalMemory(*S, F);
      }
    }
  }
  
  void storeVariableInLocalMemory(clang::VarDecl* V) const {
    HIPSYCL_DEBUG_INFO
                  << "AST Processing: Marking variable "
                  << V->getNameAsString()
                  << " as __shared__"
                  << std::endl;

    if (!V->hasAttr<clang::CUDASharedAttr>()) {
      V->addAttr(clang::CUDASharedAttr::CreateImplicit(
          Instance.getASTContext()));
      V->setStorageClass(clang::SC_Static);
    }
  }

};




class FrontendASTConsumer : public clang::ASTConsumer {
  
  FrontendASTVisitor Visitor;
  clang::CompilerInstance& Instance;
  
public:
  FrontendASTConsumer(clang::CompilerInstance &I)
      : Visitor{I}, Instance{I}
  {
    CompilationStateManager::get().reset();
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override {
    for (auto&& D : DG)
      Visitor.TraverseDecl(D);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    
    CompilationStateManager::getASTPassState().setDeviceCompilation(
        Instance.getSema().getLangOpts().CUDAIsDevice);

    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __device__ ****** " << std::endl;
    else
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __host__ ****** " << std::endl;

    Visitor.applyAttributes();

    // The following part is absolutely crucial:
    //
    // clang works roughly like this when building and processing the AST:
    // while(!not done){
    //   DeclGroupRef DG = parseNextDeclGroup()
    //   foreach(ASTConsumer C){
    //     C->HandleTopLevelDecl(DG);
    //   }
    // }
    // }
    // foreach(ASTConsumer C){
    //   C->HandleTranslationUnit(TU);
    // }
    //
    // The BackendConsumers which take care of emitting IR code
    // already emit in HandleTopLevelDecl().
    // This means that, since we have only made attribute changes in
    // HandleTranslationUnit(), all code has already been emitted without taking
    // into account our changes. In particular, since functions used in SYCL
    // kernels hadn't yet been marked as __device__ at this point, none of them
    // actually got emitted. To fix this, we let all registered ASTConsumers run
    // their HandleTopLevelDecl() over the functions where we have added
    // attributes. Since we do not implement HandleTopLevelDecl(), the only
    // consumers affected are the BackendConsumers which will then generate the
    // required IR for device code.
    clang::ASTConsumer& C = Instance.getASTConsumer();
    if(clang::isa<clang::MultiplexConsumer>(&C))
    {
      clang::MultiplexConsumer& MC = static_cast<clang::MultiplexConsumer&>(C);
      if(CompilationStateManager::getASTPassState().isDeviceCompilation()){
      
        for (clang::FunctionDecl *HDFunction :
            Visitor.getMarkedHostDeviceFunctions()) {
          clang::DeclGroupRef DG{HDFunction};

          MC.HandleTopLevelDecl(DG);
        }
      }
      // We need to reemit kernels both in host and device passes
      // to make sure the right stubs are generated
      for(clang::FunctionDecl* Kernel : Visitor.getKernels()){
          clang::DeclGroupRef DG{Kernel};

          MC.HandleTopLevelDecl(DG);
      }
    }
  }
};

}
}

#endif
