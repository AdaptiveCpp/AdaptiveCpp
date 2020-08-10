/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include <cassert>

#include "hipSYCL/runtime/dag_submitted_ops.hpp"
#include "hipSYCL/runtime/dag_node.hpp"

namespace hipsycl {
namespace rt {

namespace {

void erase_completed_nodes(std::vector<dag_node_ptr> &ops) {
  ops.erase(std::remove_if(
                ops.begin(), ops.end(),
                [&](dag_node_ptr node) -> bool { return node->is_complete(); }),
            ops.end());
}
}

void dag_submitted_ops::update_with_submission(const dag_interpreter &dag) {
  std::lock_guard lock{_lock};

  erase_completed_nodes(_ops);
  
  dag.for_each_effective_node([this](dag_node_ptr node){
    assert(node->is_submitted());
    _ops.push_back(node);
  });
}

void dag_submitted_ops::update_with_submission(dag_node_ptr single_node) {
  std::lock_guard lock{_lock};

  erase_completed_nodes(_ops);

  assert(single_node->is_submitted());
  _ops.push_back(single_node);
}

void dag_submitted_ops::wait_for_all() {
  std::lock_guard lock{_lock};

  for(dag_node_ptr node : _ops) {
    assert(node->is_submitted());
    node->wait();
  }
}

}
}