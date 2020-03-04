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


#include "CL/sycl/detail/scheduling/util.hpp"
#include "CL/sycl/detail/scheduling/operations.hpp"
#include "CL/sycl/detail/scheduling/dag_builder.hpp"
#include "CL/sycl/exception.hpp"


namespace cl {
namespace sycl {
namespace detail {


class kernel_operation;
class memcpy_operation;
class prefetch_operation;

dag_builder::dag_builder(const execution_hints& dag_hints)
: _hints{dag_hints}
{}


dag_node_ptr dag_builder::build_node(std::unique_ptr<operation> op,
                                     const requirements_list& requirements,
                                     const execution_hints& hints)
{
  // Start with global hints for this dag builder
  execution_hints operation_hints = _hints;
  // merge in any hints specific to this operation,
  // overwriting duplicates
  operation_hints.overwrite_with(hints);

  // Calculate additional requirements:
  // Iterate over all requirements and look for conflicting accesses
  for(auto node : requirements.get()){
    auto* req = cast<requirement>(node->get_operation());

    if(req->is_memory_requirement()){
      auto* mem_req = cast<memory_requirement>(req);

      // Make sure to also add the requirements of the operation to the DAG
      _current_dag.add_memory_requirement(node);

      if(mem_req->is_image_requirement())
        assert(false && "dag_builder: Image requirements are unimplemented");
      else {
        auto* buff_req = cast<buffer_memory_requirement>(req);

        data_user_tracker& user_tracker = buff_req->get_data_region()->get_users();

        for(auto user = user_tracker.users_begin(); 
            user != user_tracker.users_end(); 
            ++user) 
        {
          
          if(is_conflicting_access(mem_req, *user))
          {
            // No reason to take a dependency into account that is alreay completed
            if(!user->user->is_complete())
              node->add_requirement(user->user);
          }
        }
      }
    }
  }

  auto operation_node = std::make_shared<dag_node>(
      operation_hints, requirements.get(), std::move(op));

  return operation_node;
}


dag_node_ptr dag_builder::add_kernel(std::unique_ptr<operation> op,
                                    const requirements_list& requirements,
                                    const execution_hints& hints)
{
  assert_is<kernel_operation>(op.get());

  std::lock_guard<std::mutex> lock{_mutex};
  
  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_kernel(node);

  return node;
}

dag_node_ptr dag_builder::add_memcpy(std::unique_ptr<operation> op,
                                    const requirements_list& requirements,
                                    const execution_hints& hints)
{
  assert_is<memcpy_operation>(op.get());

  std::lock_guard<std::mutex> lock{_mutex};
  
  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_memcpy(node);

  return node;
}

dag_node_ptr dag_builder::add_fill(std::unique_ptr<operation> op,
                                  const requirements_list& requirements,
                                  const execution_hints& hints)
{
  assert_is<kernel_operation>(op.get());

  std::lock_guard<std::mutex> lock{_mutex};
  
  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_fill(node);

  return node;
}

dag_node_ptr dag_builder::add_prefetch(std::unique_ptr<operation> op,
                                      const requirements_list& requirements,
                                      const execution_hints& hints)
{
  assert_is<prefetch_operation>(op.get());

  std::lock_guard<std::mutex> lock{_mutex};
  
  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_prefetch(node);

  return node;
}

dag dag_builder::finish_and_reset()
{
  std::lock_guard<std::mutex> lock{_mutex};

  dag final_dag = _current_dag;
  _current_dag = dag{};

  return final_dag;
}

bool dag_builder::is_conflicting_access(
    const memory_requirement* mem_req, const data_user& user) const
{
  if (mem_req->get_access_mode() == access::mode::read &&
      user.mode == access::mode::read)
    return false;

  // Check if the page ranges do not intersect
  // need to determine page range
  return mem_req->intersects_with(user);
}


}
}
}