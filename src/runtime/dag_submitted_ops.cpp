/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include <cassert>

#include "hipSYCL/runtime/dag_submitted_ops.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"

namespace hipsycl {
namespace rt {

namespace {

void erase_known_completed_nodes(std::vector<dag_node_ptr> &ops) {
  ops.erase(std::remove_if(ops.begin(), ops.end(),
                           [&](dag_node_ptr node) -> bool {
                             return node->is_known_complete();
                           }),
            ops.end());
}
}

dag_submitted_ops::~dag_submitted_ops() {
  this->purge_known_completed();
}

void dag_submitted_ops::copy_node_list(std::vector<dag_node_ptr>& out) const {
  std::lock_guard lock{_lock};
  out = _ops;
}

void dag_submitted_ops::purge_known_completed() {
  std::lock_guard lock{_lock};

  erase_known_completed_nodes(_ops);
}

std::size_t dag_submitted_ops::get_num_nodes() const {
  std::lock_guard lock{_lock};
  return _ops.size();
}

void dag_submitted_ops::async_wait_and_unregister() {
    
    // If the updater thread is currently not busy with anything,
    // create a new task that waits and purges all nodes starting
    // from the most recent node
    if(_updater_thread.queue_size() == 0) {
      _updater_thread([this](){
        std::vector<dag_node_ptr> gc_node_list;
        this->copy_node_list(gc_node_list);

        for(int i = gc_node_list.size() - 1; i >= 0; --i)
          gc_node_list[i]->wait();
        
        this->purge_known_completed();
      });
    }
}

void dag_submitted_ops::update_with_submission(dag_node_ptr single_node) {
  std::lock_guard lock{_lock};

  assert(single_node->is_submitted());
  _ops.push_back(single_node);
}

void dag_submitted_ops::wait_for_all() {
  std::vector<dag_node_ptr> current_ops;
  {
    std::lock_guard lock{_lock};
    current_ops = _ops;
  }
  
  for(dag_node_ptr node : current_ops) {
    assert(node->is_submitted());
    node->wait();
  }
}

void dag_submitted_ops::wait_for_group(std::size_t node_group) {
  HIPSYCL_DEBUG_INFO << "dag_submitted_ops: Waiting for node group "
                     << node_group << std::endl;
  
  std::vector<dag_node_ptr> current_ops;
  {
    std::lock_guard lock{_lock};  
    current_ops = _ops;
  }

  // Iterate in reverse order over the nodes, since current_ops
  // will contain the nodes in submission order.
  // This means that the last nodes will be the newest. Waiting
  // on them first might turn waits on earlier nodes into no-ops
  // since dag_node::wait() also marks all requirements recursively
  // as complete.
  for(int i = current_ops.size() - 1; i >= 0; --i) {
    const dag_node_ptr& node = current_ops[i];
    assert(node->is_submitted());
    if (const hints::node_group *g =
            node->get_execution_hints().get_hint<hints::node_group>()) {
      if (g->get_id() == node_group) {
        HIPSYCL_DEBUG_INFO
            << "dag_submitted_ops: Waiting for node group; current node: "
            << node.get() << std::endl;
        node->wait();
      }
    }
  }
}

node_list_t dag_submitted_ops::get_group(std::size_t node_group) {
  
  node_list_t ops;
  {
    std::lock_guard lock{_lock};
    for(dag_node_ptr node : _ops) {
      assert(node->is_submitted());
      if (const hints::node_group *g =
              node->get_execution_hints().get_hint<hints::node_group>()) {
        if (g->get_id() == node_group) {
          ops.push_back(node);
        }
      }
    }
  }
  return ops;
}

bool dag_submitted_ops::contains_node(dag_node_ptr node) const {
  std::lock_guard lock{_lock};

  for(dag_node_ptr n : _ops) {
    if(n == node)
      return true;
  }
  return false;
}

}
}
