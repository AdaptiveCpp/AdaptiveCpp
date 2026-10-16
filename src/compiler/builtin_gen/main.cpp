#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include <iostream>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;



class BuiltinGenConsumer : public ASTConsumer {
  std::string HppPath;
  std::string CppPath;
public:
  BuiltinGenConsumer(std::string Hpp, std::string Cpp) : HppPath(Hpp), CppPath(Cpp) {}

  std::string sanitizeType(std::string T) {
    size_t pos;
    while ((pos = T.find("_Bool")) != std::string::npos) {
      T.replace(pos, 5, "bool");
    }
    while ((pos = T.find(" __attribute__((ext_vector_type(")) != std::string::npos) {
      size_t endPos = T.find(")))", pos);
      if (endPos != std::string::npos) {
        std::string num = T.substr(pos + 32, endPos - (pos + 32));
        std::string base = T.substr(0, pos);
        // clean base type, e.g., "unsigned int" -> "uint"
        std::string baseName = base;
        while((pos = baseName.find(" ")) != std::string::npos) {
           baseName.replace(pos, 1, "_");
        }
        std::string newType = "__acpp_vec_" + baseName + "_" + num;
        T = newType + T.substr(endPos + 3);
      }
    }
    return T;
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    std::error_code EC1, EC2;
    llvm::raw_fd_ostream HppFile(HppPath, EC1, llvm::sys::fs::OF_None);
    llvm::raw_fd_ostream CppFile(CppPath, EC2, llvm::sys::fs::OF_None);

    if (EC1 || EC2) {
      llvm::errs() << "Error opening output files.\n";
      return;
    }

    HppFile << "// Auto-generated AMDGPU Builtins Declarations\n"
            << "#pragma once\n\n"
            << "#pragma clang diagnostic push\n"
            << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n\n"
            << "typedef float __acpp_vec_float_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef float __acpp_vec_float_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef float __acpp_vec_float_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef float __acpp_vec_float_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef float __acpp_vec_float_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef float __acpp_vec_float_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef double __acpp_vec_double_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef double __acpp_vec_double_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef double __acpp_vec_double_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef double __acpp_vec_double_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef double __acpp_vec_double_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef int __acpp_vec_int_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef int __acpp_vec_int_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef int __acpp_vec_int_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef int __acpp_vec_int_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef int __acpp_vec_int_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef int __acpp_vec_int_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef short __acpp_vec_short_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef short __acpp_vec_short_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef short __acpp_vec_short_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef short __acpp_vec_short_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef short __acpp_vec_short_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef __fp16 __acpp_vec___fp16_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef __fp16 __acpp_vec___fp16_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef __fp16 __acpp_vec___fp16_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef __fp16 __acpp_vec___fp16_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef __fp16 __acpp_vec___fp16_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef __fp16 __acpp_vec___fp16_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef long long __acpp_vec_long_long_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef long long __acpp_vec_long_long_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef long long __acpp_vec_long_long_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef long long __acpp_vec_long_long_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef long long __acpp_vec_long_long_16 __attribute__((ext_vector_type(16)));\n"
            << "extern \"C\" {\n";

    CppFile << "// Auto-generated AMDGPU Builtins Implementations\n\n"
            << "#pragma clang diagnostic push\n"
            << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n\n"
            << "#include \"hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_auto_builtins.hpp\"\n\n"
            << "extern \"C\" {\n";

    // Initialize builtins so they populate the IdentifierTable
    Ctx.BuiltinInfo.initializeBuiltins(Ctx.Idents, Ctx.getLangOpts());
    
    std::vector<unsigned> BuiltinIDs;
    for (auto it = Ctx.Idents.begin(); it != Ctx.Idents.end(); ++it) {
      unsigned ID = it->getValue()->getBuiltinID();
      if (ID >= Builtin::FirstTSBuiltin) {
        std::string Name = it->getKey().str();
        if (Name.find("amdgcn") != std::string::npos || Name.find("amdgpu") != std::string::npos) {
          BuiltinIDs.push_back(ID);
        }
      }
    }
    // Sort to ensure deterministic generation
    std::sort(BuiltinIDs.begin(), BuiltinIDs.end(), [&](unsigned a, unsigned b) {
      return Ctx.BuiltinInfo.getName(a) < Ctx.BuiltinInfo.getName(b);
    });

    for (unsigned ID : BuiltinIDs) {
      std::string Name(Ctx.BuiltinInfo.getName(ID));
      const char* Features = Ctx.BuiltinInfo.getRequiredFeatures(ID);
      
      if (Name.find("atomic_inc") != std::string::npos ||
          Name.find("atomic_dec") != std::string::npos ||
          Name.find("fence") != std::string::npos ||
          Name.find("div_scale") != std::string::npos ||
          Name.find("interp") != std::string::npos ||
          Name.find("buffer_rsrc") != std::string::npos ||
          Name.find("r600_") != std::string::npos) {
        continue;
      }

      ASTContext::GetBuiltinTypeError Error;
      unsigned IntegerConstantArgs = 0;
      QualType FuncType = Ctx.GetBuiltinType(ID, Error, &IntegerConstantArgs);
      
      if (Error == ASTContext::GE_None && !FuncType.isNull()) {
        const auto* FPT = FuncType->getAs<FunctionProtoType>();
        if (!FPT) continue;
        
        // Build the parameters string
        std::string ParamsStr;
        llvm::raw_string_ostream ParamsOS(ParamsStr);
        for (unsigned i = 0; i < FPT->getNumParams(); ++i) {
          if (i > 0) ParamsOS << ", ";
          ParamsOS << sanitizeType(FPT->getParamType(i).getAsString()) << " arg" << i;
        }

        std::string Sig = llvm::formatv("{0} __acpp_{1}({2})", 
            sanitizeType(FPT->getReturnType().getAsString()), Name, ParamsOS.str()).str();

        if (Sig.find("__fp16") != std::string::npos || Sig.find("__amdgpu_buffer_rsrc_t") != std::string::npos) {
          continue;
        }

        // Write declaration to .hpp
        HppFile << Sig << ";\n";
        
        // Build the arguments string
        std::string ArgsStr;
        llvm::raw_string_ostream ArgsOS(ArgsStr);
        for (unsigned i = 0; i < FPT->getNumParams(); ++i) {
          if (i > 0) ArgsOS << ", ";
          if (IntegerConstantArgs & (1 << i)) {
            ArgsOS << "1";
          } else {
            ArgsOS << "arg" << i;
          }
        }

        // Write definition to .cpp
        CppFile << "__attribute__((always_inline))\n";
        if (Features && Features[0] != '\0') {
            CppFile << llvm::formatv("__attribute__((target(\"{0}\")))\n", Features);
        }
        
        CppFile << llvm::formatv("{0} {\n  {1}{2}({3});\n}\n\n",
            Sig,
            FPT->getReturnType()->isVoidType() ? "" : "return ",
            Name,
            ArgsOS.str());
      }
    }
    
    HppFile << "}\n#pragma clang diagnostic pop\n";
    CppFile << "}\n#pragma clang diagnostic pop\n";
  }
};

class BuiltinGenAction : public ASTFrontendAction {
  std::string HppPath;
  std::string CppPath;
public:
  BuiltinGenAction(std::string Hpp, std::string Cpp) : HppPath(Hpp), CppPath(Cpp) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    return std::make_unique<BuiltinGenConsumer>(HppPath, CppPath);
  }
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <output_hpp_path> <output_cpp_path>\n";
    return 1;
  }
  
  std::string HppPath = argv[1];
  std::string CppPath = argv[2];

  std::vector<std::string> args = {"-target", "amdgcn-amd-amdhsa", "-nogpulib", "-fsyntax-only"};
  bool success = tooling::runToolOnCodeWithArgs(std::make_unique<BuiltinGenAction>(HppPath, CppPath), "void dummy(){}", args);
  if (!success) {
    std::cerr << "runToolOnCodeWithArgs failed!\n";
    return 1;
  }
  std::cerr << "Finished successfully.\n";
  return 0;
}
