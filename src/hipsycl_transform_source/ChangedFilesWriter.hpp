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

#ifndef HIPSYCL_CHANGED_FILES_WRITER_HPP
#define HIPSYCL_CHANGED_FILES_WRITER_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Rewrite/Core/Rewriter.h>

namespace hipsycl {
namespace transform {

class CommentParser;

class ChangedFilesWriter
{
public:
  ChangedFilesWriter(clang::HeaderSearch& search,
                     clang::SourceManager& sourceMgr,
                     clang::Rewriter& r,
                     const std::string& mainOutputFilename);

  void run(const std::string& basePath);

private:
  clang::HeaderSearch& _search;
  clang::SourceManager& _sourceMgr;
  clang::Rewriter& _rewriter;
  std::string _mainFilename;

  typedef unsigned int FileIDHash;

  using includeFileReferences = std::pair<clang::SourceLocation, const clang::FileEntry*>;

  std::unordered_map<const clang::FileEntry*, clang::FileID> _rewrittenFiles;
  std::unordered_map<const clang::FileEntry*, std::vector<includeFileReferences>>
        _rewrittenIncludes;

  std::string getNewFilename(const clang::FileEntry*) const;

  bool isFilePlannedForRewrite(clang::FileID file) const;

  void markDependentFilesForRewrite(clang::FileID file);


  void processFile(const std::string& basePath,
                   const clang::FileEntry* entry,
                   clang::FileID id);

  bool lineIsInclude(const std::string& line,
                     const CommentParser& comments,
                     std::size_t lineOffset,
                     std::size_t& includeDirectiveEnd) const;

  bool findIncludeFilenameSpecification(const std::string& line,
                                        const CommentParser& comments,
                                        const std::size_t lineOffset,
                                        const std::size_t searchStart,
                                        std::string& include,
                                        bool& isAngled) const;

  std::string generateIncludeDirective(const std::string& includeFile,
                                       bool isAngled) const;

  bool mustIncludeBeRewritten(const clang::FileEntry* file,
                              std::size_t line,
                              const clang::FileEntry*& includedFile) const;

  void writeFile(const std::string& basePath,
                 const std::string& filename,
                 const std::string& content) const;
};

}
}

#endif
