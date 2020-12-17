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


#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/dag_builder.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/instrumentation.hpp"

#include <mutex>

// TODO: Implement the following optimizations:
// a) Remove unnecessary requirements, i.e. when a requirement has a dependency
// on a node that is also satisfied by another node:
//  node -> req1 -> node 2
//       -> node2
// b) Reorder requirements such that larger accesses come first. This will cause
// later requirements to potentially be optimized away entirely.


namespace hipsycl {
namespace rt {

namespace {

// Add this node to the data users of the memory region of the specified
// requirement
void add_to_data_users(dag_node_ptr node, memory_requirement *mem_req) {
  assert(mem_req);
  if (mem_req->is_buffer_requirement()) {

    auto &data_users = cast<buffer_memory_requirement>(mem_req)
                           ->get_data_region()
                           ->get_users();

    if (!data_users.has_user(node)) {
      data_users.add_user(
          node, mem_req->get_access_mode(), mem_req->get_access_target(),
          mem_req->get_access_offset3d(), mem_req->get_access_range3d());
    }
  } else
    assert(false && "dag: Image requirements are not yet implemented");
}

// Add this node to the data users of the memory regions
// referenced in the requirements
void add_to_data_users(dag_node_ptr node, const requirements_list& reqs)
{
  for(dag_node_ptr req : reqs.get()){
    if(req->get_operation()->is_requirement()){
      auto* mem_req = cast<memory_requirement>(req->get_operation());

      add_to_data_users(node, mem_req);
    }
  }
}

}



class kernel_operation;
class memcpy_operation;
class prefetch_operation;

dag_builder::dag_builder(){}


dag_node_ptr dag_builder::build_node(std::unique_ptr<operation> op,
                                     const requirements_list& requirements,
                                     const execution_hints& hints)
{
  assert(op);

  // Calculate additional requirements:
  // Iterate over all requirements and look for conflicting accesses

  // For a given requirement, checks for conflicts and adds any
  // conflicting operations as dependencies
  auto add_conflicts_as_requirements = [&](dag_node_ptr req_node){
    if(req_node->get_operation()->is_requirement()){
      auto* req = cast<requirement>(req_node->get_operation());

      if(req->is_memory_requirement()){
        auto* mem_req = cast<memory_requirement>(req);

        if(mem_req->is_image_requirement())
          assert(false && "dag_builder: Image requirements are unimplemented");
        else {
          auto* buff_req = cast<buffer_memory_requirement>(req);

          data_user_tracker &user_tracker =
              buff_req->get_data_region()->get_users();

          user_tracker.for_each_user([&](data_user &user) {
            auto user_ptr = user.user.lock();
            if(user_ptr && is_conflicting_access(mem_req, user))
            {
              // No reason to take a dependency into account that is alreay completed
              if(!user_ptr->is_complete())
                req_node->add_requirement(user_ptr);
            }
          });
          
        }
      }
    }
  };

  auto operation_node = std::make_shared<dag_node>(
      hints, requirements.get(), std::move(op));
  
  if(operation_node->get_operation()->is_requirement()) {
    add_conflicts_as_requirements(operation_node);
    // if this is an explicit requirement, we need to add *this*
    // operation to the users of the requirement it refers to.
    requirement* req = cast<requirement>(operation_node->get_operation());
    if(req->is_memory_requirement())
      add_to_data_users(operation_node, cast<memory_requirement>(
                                            operation_node->get_operation()));
  }

  for (auto node : operation_node->get_requirements())
      add_conflicts_as_requirements(node);
  
  add_to_data_users(operation_node, requirements);

  return operation_node;
}

dag_node_ptr
dag_builder::add_command_group(std::unique_ptr<operation> op,
                               const requirements_list &requirements,
                               const execution_hints &hints)
{
  assert(op);

  // Since requirements may be optimized away, requirements, even if issued explicitly,
  // currently cannot be profiled
  if (!op->is_requirement() && hints.has_hint<hints::enable_profiling>()) {
    op->get_instrumentations().instrument<rt::timestamp_profiler>();
  }

  std::lock_guard<std::mutex> lock{_mutex};

  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_command_group(node);

  return node;
}

dag_node_ptr dag_builder::add_kernel(std::unique_ptr<operation> op,
                                     const requirements_list &requirements,
                                     const execution_hints &hints)
{
  assert_is<kernel_operation>(op.get());
  return add_command_group(std::move(op), requirements, hints);
}

dag_node_ptr dag_builder::add_memcpy(std::unique_ptr<operation> op,
                                     const requirements_list &requirements,
                                     const execution_hints &hints)
{
  assert_is<memcpy_operation>(op.get());
  return add_command_group(std::move(op), requirements, hints);
}

dag_node_ptr dag_builder::add_fill(std::unique_ptr<operation> op,
                                   const requirements_list &requirements,
                                   const execution_hints &hints)
{
  assert_is<kernel_operation>(op.get());
  return add_command_group(std::move(op), requirements, hints);
}

dag_node_ptr dag_builder::add_prefetch(std::unique_ptr<operation> op,
                                      const requirements_list& requirements,
                                      const execution_hints& hints)
{
  assert_is<prefetch_operation>(op.get());
  return add_command_group(std::move(op), requirements, hints);
}

dag_node_ptr dag_builder::add_memset(std::unique_ptr<operation> op,
                                      const requirements_list& requirements,
                                      const execution_hints& hints)
{
  assert_is<memset_operation>(op.get());
  return add_command_group(std::move(op), requirements, hints);
}

dag_node_ptr dag_builder::add_explicit_mem_requirement(
    std::unique_ptr<operation> req,
    const requirements_list &requirements, const execution_hints &hints)
{
  assert_is<memory_requirement>(req.get());
  return add_command_group(std::move(req), requirements, hints);
}

dag dag_builder::finish_and_reset()
{
  std::lock_guard<std::mutex> lock{_mutex};

  dag final_dag = _current_dag;
  _current_dag = dag{};

  final_dag.for_each_node([](dag_node_ptr node) {
    HIPSYCL_DEBUG_INFO << "dag_builder: DAG contains operation: "
                       << dump(node->get_operation()) << std::endl;
    for (dag_node_ptr req : node->get_requirements()) {
      HIPSYCL_DEBUG_INFO << "    --> requires: " << dump(req->get_operation()) << std::endl;
    }
  });

  return final_dag;
}

bool dag_builder::is_conflicting_access(
    const memory_requirement* mem_req, const data_user& user) const
{
  if (mem_req->get_access_mode() == sycl::access::mode::read &&
      user.mode == sycl::access::mode::read)
    return false;

  // TODO Check if the page ranges do not intersect
  // need to determine page range
  return mem_req->intersects_with(user);
}

std::size_t dag_builder::get_current_dag_size() const
{
  std::lock_guard<std::mutex> lock{_mutex};
  return _current_dag.num_nodes();
}

}
}
