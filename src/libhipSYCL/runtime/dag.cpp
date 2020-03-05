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


#include <algorithm>

#include "hipSYCL/runtime/util.hpp"
#include "CL/sycl/exception.hpp"
#include "hipSYCL/runtime/dag.hpp"

namespace cl {
namespace sycl {
namespace detail {

// Kernel execution
void dag::add_kernel(dag_node_ptr kernel)
{
  _kernels.push_back(kernel);
}

// Explicit copy operation - must be executed
void dag::add_memcpy(dag_node_ptr memcpy)
{
  _memcopies.push_back(memcpy);
}

// Explicit fill
void dag::add_fill(dag_node_ptr fill)
{
  _fills.push_back(fill);
}

// USM/SVM prefetch
void dag::add_prefetch(dag_node_ptr prefetch)
{
  _prefetches.push_back(prefetch);
}

// memory requirements. DAGs that have requirements are not
// executable until all requirements have been translated
// into actual operations or removed.
void dag::add_memory_requirement(dag_node_ptr requirement)
{
  _memory_requirements.push_back(requirement);
}

const std::vector<dag_node_ptr>& 
dag::get_kernels() const
{
  return _kernels;
}

const std::vector<dag_node_ptr>& 
dag::get_memory_requirements() const
{
  return _memory_requirements;
}

// Registers the dependencies of all requirements in this DAG with
// the dependency trackers associated with the data buffers
void dag::commit_node_dependencies()
{
  this->commit_dependencies(_kernels);
  this->commit_dependencies(_memcopies);
  this->commit_dependencies(_fills);
  this->commit_dependencies(_prefetches);
}

bool dag::is_requirement_from_this_dag(const dag_node_ptr& node) const
{
  assert_is<requirement>(node->get_operation());

  return std::find(_memory_requirements.begin(), 
                  _memory_requirements.end(), node) 
                  != _memory_requirements.end();
}

void dag::commit_dependencies(const std::vector<dag_node_ptr>& nodes)
{
  for(dag_node_ptr node : nodes)
  {
    for(dag_node_ptr req : node->get_requirements())
    {
      // Ignore requirements that may have already
      // been processed by a previous DAG
      if(is_requirement_from_this_dag(req))
      {
        auto* mem_req = cast<memory_requirement>(req->get_operation());

        if(!mem_req->is_image_requirement())
        {
          auto& data_users = cast<buffer_memory_requirement>(mem_req)->
              get_data_region()->get_users();

          if(data_users.find_user(req) == data_users.users_end())
            data_users.add_user(node, 
                                mem_req->get_access_mode(),
                                mem_req->get_access_target(),
                                mem_req->get_access_offset3d(),
                                mem_req->get_access_range3d());
        }
        else
          throw unimplemented{"dag: Image requirements are not yet implemented"};
      }
    }
  }
}

bool dag::contains_node(dag_node_ptr node) const
{
  if(std::find(_kernels.begin(), _kernels.end(), node) != _kernels.end())
    return true;
  if(std::find(_memcopies.begin(), _memcopies.end(), node) != _memcopies.end())
    return true;
  if(std::find(_fills.begin(), _fills.end(), node) != _fills.end())
    return true;
  if(std::find(_prefetches.begin(), _prefetches.end(), node) != _prefetches.end())
    return true;
  return false;
}

void dag::for_each_node(std::function<void(dag_node_ptr)> handler) const
{
  std::for_each(_kernels.begin(), 
                _kernels.end(), handler);
  std::for_each(_memcopies.begin(), 
                _memcopies.end(), handler);
  std::for_each(_fills.begin(), 
                _fills.end(), handler);
  std::for_each(_prefetches.begin(), 
                _prefetches.end(), handler);
}

}
}
}