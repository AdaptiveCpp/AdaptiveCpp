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
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include "ChangedFilesWriter.hpp"
#include <stdexcept>
#include <iostream>

namespace hipsycl {
namespace transform {

class CommentParser
{
public:
  CommentParser(const std::string& code)
    : _code{code}
  {
    if(!code.empty())
    {
      for(std::size_t i = 0; i < code.size() - 1; ++i)
      {
        if(code[i] == '/' && code[i+1] == '*')
        {
          _commentsBegin.push_back(i);
          for(; i+1 < code.size() &&
              (code[i] != '*' || code[i+1] != '/'); ++i)
            ;
          _commentsEnd.push_back(i+1);
        }
        else if(code[i] == '/' && code[i+1] == '/')
        {
          _commentsBegin.push_back(i);

          for(; i < code.size() && code[i] != '\n'; ++i)
            ;

          _commentsEnd.push_back(i);

        }
      }
    }
  }

  bool isComment(std::size_t pos) const
  {
    if(pos >= _code.size())
      return false;

    assert(_commentsBegin.size() == _commentsEnd.size());

    for(std::size_t i = 0; i < _commentsBegin.size(); ++i)
    {
      if(_commentsBegin[i] <= pos && _commentsEnd[i] >= pos)
        return true;
    }
    return false;
  }

private:
  std::string _code;
  std::vector<std::size_t> _commentsBegin;
  std::vector<std::size_t> _commentsEnd;
};


ChangedFilesWriter::ChangedFilesWriter(clang::HeaderSearch& search,
                                       clang::SourceManager& sourceMgr,
                                       clang::Rewriter& r,
                                       const std::string& mainOutputFilename)
  : _search{search},
    _sourceMgr{sourceMgr},
    _rewriter{r},
    _mainFilename{mainOutputFilename}
{
  assert(!_mainFilename.empty());
  assert(_mainFilename.find("/") == std::string::npos);
}

std::string
ChangedFilesWriter::getNewFilename(const clang::FileEntry* entry) const
{
  if(_sourceMgr.getFileEntryForID(_sourceMgr.getMainFileID()) == entry)
    return _mainFilename;

  return "hipsycl_"+std::to_string(entry->getUID())+".hpp";
}

void ChangedFilesWriter::run(const std::string& basePath)
{
  _rewrittenFiles.clear();
  _rewrittenIncludes.clear();

  // Mark all modified files and all files depending on he modified files for rewrite
  for(auto buff = _rewriter.buffer_begin();
      buff != _rewriter.buffer_end();
      ++buff)
  {
    clang::FileID id = buff->first;
    markDependentFilesForRewrite(id);
  }

  for(auto modifiedFile : _rewrittenFiles)
  {
    this->processFile(basePath, modifiedFile.first, modifiedFile.second);
  }
}


bool ChangedFilesWriter::isFilePlannedForRewrite(clang::FileID file) const
{
  return _rewrittenFiles.find(_sourceMgr.getFileEntryForID(file)) != _rewrittenFiles.end();
}

void ChangedFilesWriter::markDependentFilesForRewrite(clang::FileID file)
{
  _rewrittenFiles[_sourceMgr.getFileEntryForID(file)] = file;

  clang::SourceLocation includeLoc = _sourceMgr.getIncludeLoc(file);
  if(includeLoc.isValid())
  {
    clang::FileID includingID = _sourceMgr.getFileID(includeLoc);

    // Avoid cycles by only starting recursion if the file hasn't been processed
    // yet
    if(!isFilePlannedForRewrite(includingID))
    {
      _rewrittenFiles[_sourceMgr.getFileEntryForID(includingID)] = includingID;
      _rewrittenIncludes[_sourceMgr.getFileEntryForID(includingID)].push_back(
            std::make_pair(includeLoc, _sourceMgr.getFileEntryForID(file)));

      markDependentFilesForRewrite(includingID);
    }
  }

}


std::string
ChangedFilesWriter::generateIncludeDirective(const std::string& includeFile,
                                             bool isAngled) const
{
  if(isAngled)
    return "#include <" + includeFile + ">";
  else
    return "#include \"" + includeFile + "\"";
}


bool
ChangedFilesWriter::mustIncludeBeRewritten(const clang::FileEntry* file,
                                          std::size_t line,
                                          const clang::FileEntry*& includedFile) const
{
  auto it = _rewrittenIncludes.find(file);

  if(it == _rewrittenIncludes.end())
    return false;

  for(auto entry : it->second)
  {
    clang::SourceLocation loc = entry.first;
    if(loc.isValid())
    {
      if(unsigned retrievedLine = _sourceMgr.getSpellingLineNumber(loc))
      {
        if(retrievedLine == line)
        {
          includedFile = entry.second;
          return true;
        }
      }
    }
  }

  return false;
}


void ChangedFilesWriter::writeFile(const std::string& basePath,
                                   const std::string& filename,
                                   const std::string& content) const
{
  std::string fullFilename = basePath;

  if(!fullFilename.empty())
  {
    if(fullFilename[basePath.size()-1] != '/')
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
      if(fullFilename[basePath.size()-1] != '\\')
#endif
      fullFilename += '/';
    }
  }
  fullFilename += filename;

  std::ofstream outputStream{fullFilename.c_str(), std::ios::trunc};
  if(outputStream.is_open())
  {
    outputStream << content;
    outputStream.close();
  }
  else
  {
    throw std::runtime_error{"Could not open file "+fullFilename+" for writing."};
  }
}

void ChangedFilesWriter::processFile(const std::string& basePath,
                                     const clang::FileEntry* entry,
                                     clang::FileID id)
{
  std::string fileContent;
  llvm::raw_string_ostream outputStream{fileContent};

  _rewriter.getEditBuffer(id).write(outputStream);
  outputStream.flush();

  CommentParser comments{fileContent};

  std::string rewrittenFileContent =
      "#line 1 \""+entry->getName().str()+"\"\n";

  std::size_t lineId = 1;

  std::string line;
  std::istringstream input{fileContent};

  std::size_t lineOffset = 0;

  while (std::getline(input, line)) {

    std::size_t includeDirectiveEnd;
    bool lineStartsWithInclude = lineIsInclude(line,
                                               comments,
                                               lineOffset,
                                               includeDirectiveEnd);
    bool isIncludeRewritten = false;

    if(lineStartsWithInclude)
    {

      const clang::FileEntry* includedFile = nullptr;
      if(mustIncludeBeRewritten(entry,lineId,includedFile))
      {
        if(includedFile)
        {
          if(_rewrittenFiles.find(includedFile) !=
             _rewrittenFiles.end())
          {
            // Use new name for file
            std::string newIncludeName = getNewFilename(includedFile);
            rewrittenFileContent += "#include \"";
            rewrittenFileContent += newIncludeName;
            rewrittenFileContent += "\"\n";
            isIncludeRewritten = true;
          }
        }
      }
    }

    if(!isIncludeRewritten)
    {
      rewrittenFileContent += line;
      rewrittenFileContent += '\n';
    }

    // +1 for the \n character that is discarded by std::getline
    lineOffset += line.size() + 1;
    ++lineId;
  }
  writeFile(basePath, getNewFilename(entry), rewrittenFileContent);
}

bool ChangedFilesWriter::lineIsInclude(const std::string& line,
                                       const CommentParser& comments,
                                       std::size_t lineOffset,
                                       std::size_t& includeDirectiveEnd) const
{
  bool foundHash = false;

  const std::string includeDirective = "include";

  for(std::size_t i = 0; i < line.size(); ++i)
  {
    if(!foundHash)
    {
      if(line[i] == '#')
        foundHash = true;
      else if(line[i] != ' ' && line[i] != '\t')
        return false;
    }
    else
    {
      if(line[i] != ' ' && line[i] != '\t' && !comments.isComment(lineOffset + i))
      {
        if(line.substr(i, includeDirective.length()) != includeDirective)
          return false;
        else
        {
          includeDirectiveEnd = i + includeDirective.length();
          return true;
        }
      }
    }
  }
  return false;
}

bool ChangedFilesWriter::findIncludeFilenameSpecification(
    const std::string& line,
    const CommentParser& comments,
    const std::size_t lineOffset,
    const std::size_t searchStart,
    std::string& includeFile,
    bool& isAngled) const
{
  if(line.empty())
    return false;

  includeFile = "";

  bool foundFilename = false;
  for(std::size_t i = searchStart;
      i < line.size(); ++i)
  {
    if(!foundFilename)
    {
      if(line[i] != ' ' && line[i] != '\t' &&
         !comments.isComment(i + lineOffset))
      {
        if(line[i] != '\"' && line[i] != '<')
          return false;

        isAngled = line[i] == '<';

        foundFilename = true;
      }
    }
    else
    {
      if((isAngled && line[i] == '>') ||
         (!isAngled && line[i] == '\"'))
        return true;
      else
        includeFile += line[i];
    }
  }

  return false;
}

}
}
