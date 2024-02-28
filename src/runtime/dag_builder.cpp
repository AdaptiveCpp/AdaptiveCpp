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
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/dag_builder.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/sycl/access.hpp"

#include <mutex>
#include <utility>

// TODO: Implement the following optimization:
// - Reorder requirements such that larger accesses come first. This will cause
// later requirements to potentially be optimized away entirely.


namespace hipsycl {
namespace rt {

namespace {

// Attempts to find x in node's dependency graph
bool find_dependency(const dag_node_ptr &node, const dag_node_ptr& x,
                                    int max_num_levels = 2) {
  if(max_num_levels <= 0)
    return false;

  for(auto& req : node->get_requirements()) {
    if(auto req_locked = req.lock()) {
      if(req_locked == x)
        return true;
    
      if(find_dependency(req_locked, x, max_num_levels - 1))
        return true;
    }
  }

  return false;
}

// Add this node to the data users of the memory region of the specified
// requirement
void add_to_data_users(dag_node_ptr node, memory_requirement *mem_req) {
  assert(mem_req);
  if (mem_req->is_buffer_requirement()) {

    auto &data_users = cast<buffer_memory_requirement>(mem_req)
                           ->get_data_region()
                           ->get_users();
    // This lambda is used to check whether an existing user
    // should be overwritten by the new user. This is an important
    // optimization: We KNOW that the new user is going to have
    // dependencies on conflicting accesses, so we can just
    // replace the tracking of those older accesses with the new one.
    auto replaces_user = [&](const data_user& user) -> bool {
      // Check if accessed range of new user is larger or equal.
      // If its range does not encompass all the range of the existing
      // user, we cannot remove the existing user.
      for(int i = 0; i < 3; ++i) {
        auto offset = mem_req->get_access_offset3d();
        auto range = mem_req->get_access_range3d();

        if (offset[i] > user.offset[i])
          return false;
        if (offset[i] + range[i] < user.offset[i] + user.range[i])
          return false;
      }

      bool new_user_writes = mem_req->get_access_mode() != sycl::access_mode::read;
      bool old_user_writes = user.mode != sycl::access_mode::read;

      // Write accesses always create strong dependencies, so we can
      // safely replace the old user in any case
      if(new_user_writes) {
        return true;
      }
      // A read-only access is weaker than a write-access, so we can
      // only replace if the other user is also a read-only access.
      else if(!old_user_writes){
        auto user_locked = user.user.lock();
        // If user does not exist anymore, it can be replaced.
        if(!user_locked)
          return true;
        // Replacement is only correct if the old user is part of the dependency chain
        // of the new user, since two read accesses otherwise might not have a dependency.
        if(find_dependency(node, user_locked)) {
          return true;
        }
      }

      return false;
    };

    // We need to add the user unconditionally, whether the user
    // is already registered or not. This is to cover the case
    // where we have multiple requirements (potentially with different access
    // modes or ranges) on the same operation.
    // This cannot introduce duplicate dependency edges in the DAG because
    // dag_node::add_requirement() only inserts requirements to nodes
    // that are not listed yet as requirement. 
    data_users.add_user(
        node, mem_req->get_access_mode(), mem_req->get_access_target(),
        mem_req->get_access_offset3d(), mem_req->get_access_range3d(),
        replaces_user);
  
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

dag_builder::dag_builder(runtime *rt) : _rt{rt} {}

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
              if(!user_ptr->is_known_complete())
                req_node->add_requirement(user_ptr);
            }
          });
        }
      }
    }
  };

  auto operation_node = std::make_shared<dag_node>(
      hints, requirements.get(), std::move(op), _rt);
  
  bool is_req = operation_node->get_operation()->is_requirement();

  // Do not change order between add_conflicts_as_requirements()
  // and add_to_data_users() to prevent this node ending up as a
  // requirement to itself or other cyclic requirements!
  if(is_req)
    // If we are an explicit requirement, consider conflicts not only
    // with our requirements, but with the node itself
    add_conflicts_as_requirements(operation_node);
  
  for (auto weak_node : operation_node->get_requirements()) {
    if(auto node = weak_node.lock())
      add_conflicts_as_requirements(node);
  }

  // if this is an explicit requirement, we need to add *this*
  // operation to the users of the requirement it refers to.
  if (is_req) {
    requirement *req = cast<requirement>(operation_node->get_operation());
    if (req->is_memory_requirement())
      add_to_data_users(operation_node, cast<memory_requirement>(
                                            operation_node->get_operation()));
  }
  add_to_data_users(operation_node, requirements);

  return operation_node;
}

dag_node_ptr
dag_builder::add_command_group(std::unique_ptr<operation> op,
                               const requirements_list &requirements,
                               const execution_hints &hints)
{
  assert(op);

  std::lock_guard<std::mutex> lock{_mutex};

  auto node = this->build_node(std::move(op), requirements, hints);
  _current_dag.add_command_group(node);

  return node;
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

  dag final_dag = std::exchange(_current_dag, {});

  HIPSYCL_DEBUG_INFO << "dag_builder: DAG contains operations: " << std::endl;
  int operation_index = 0;
  final_dag.for_each_node([&](dag_node_ptr node) {
    HIPSYCL_DEBUG_INFO << operation_index << ". " << dump(node->get_operation())
                       << " @node " << node.get() << std::endl;

    for (auto weak_req : node->get_requirements()) {
      if(auto req = weak_req.lock()) {
        HIPSYCL_DEBUG_INFO << "    --> requires node @" << req.get()
                          << " " << dump(req->get_operation()) << std::endl;
      }
    }

    ++operation_index;
  });

  return final_dag;
}

bool dag_builder::is_conflicting_access(
    const memory_requirement* mem_req, const data_user& user) const
{
  if (mem_req->get_access_mode() == sycl::access::mode::read &&
      user.mode == sycl::access::mode::read)
    return false;

  // Checks if the page ranges do not intersect
  return mem_req->intersects_with(user);
}

std::size_t dag_builder::get_current_dag_size() const
{
  std::lock_guard<std::mutex> lock{_mutex};
  return _current_dag.num_nodes();
}

}
}
