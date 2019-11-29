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

#include "CL/sycl/detail/scheduling/hints.hpp"
#include "CL/sycl/detail/scheduling/device_id.hpp"
#include "CL/sycl/detail/scheduling/dag_scheduler.hpp"
#include "CL/sycl/detail/scheduling/dag_enumerator.hpp"
#include "CL/sycl/detail/scheduling/dag.hpp"
#include "CL/sycl/detail/scheduling/dag_expander.hpp"


namespace cl {
namespace sycl {
namespace detail {

namespace {

struct scheduling_state
{
  std::vector<node_scheduling_annotation> scheduling_annotations;
  dag_expansion_result expansion_result;
};

}

void dag_scheduler::submit(dag* d)
{
  // Start by enumerating all nodes in the dag
  dag_enumerator enumerator{d};

  // Next, create scheduling annotations for each node
  // and extract already present information regarding
  // the target device
  std::vector<node_scheduling_annotation> scheduling_annotations(
      enumerator.get_node_index_space_size());

  std::vector<char> is_device_predetermined(
      enumerator.get_node_index_space_size(), false);

  d->for_each_node([&](dag_node_ptr node) {
    // for each node, if it comes with information on which device to execute...
    if (node->get_execution_hints().has_hint(
            execution_hint_type::bind_to_device)) {

      assert(node->get_execution_hints().has_hint(
          execution_hint_type::dag_enumeration_id));

      // ... get the node id
      std::size_t node_id = node->get_execution_hints()
                                .get_hint<hints::dag_enumeration_id>()
                                ->id();

      // remember that the node with this id is *not* free
      // to be scheduled to arbitrary devices
      is_device_predetermined[node_id] = true;
      // instead move the supplied target device to the scheduling annotations
      scheduling_annotations[node_id].set_target_device(
          node->get_execution_hints()
              .get_hint<hints::bind_to_device>()
              ->get_device_id());
    }
  });



  // Register users of buffers
  // TODO: Could we also invoke that function earlier, e.g. in the dag_builder,
  // since the nodes themselves and their dependencies shouldn't change during
  // scheduling?
  d->commit_node_dependencies();
}

} // detail
} // sycl
} // cl
