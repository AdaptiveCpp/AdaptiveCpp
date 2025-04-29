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
#ifndef HIPSYCL_DAG_BUILDER_HPP
#define HIPSYCL_DAG_BUILDER_HPP

#include "dag.hpp"
#include "dag_node.hpp"
#include "operations.hpp"
#include "hints.hpp"

#include <mutex>
#include <memory>

namespace hipsycl {
namespace rt {

class runtime;
/// Incrementally builds a dag based on operations, taking into
/// account data dependencies.
/// The resulting DAG will still contain requirements,
/// which must be transformed in a a later stage by the scheduler
///
/// Note: At any given time, there can only exist one dag_builder, otherwise
/// calculated dependencies may be incorrect!
///
/// Thread safety: Safe
class dag_builder
{
public:
  dag_builder(runtime* rt);

  dag_node_ptr add_command_group(std::unique_ptr<operation> op,
                                const requirements_list& requirements,
                                const execution_hints& hints = {});

  dag_node_ptr
  add_explicit_mem_requirement(std::unique_ptr<operation> req,
                               const requirements_list &requirements,
                               const execution_hints &hints = {});

  dag finish_and_reset();

  std::size_t get_current_dag_size() const;
private:
  bool is_conflicting_access(const memory_requirement *mem_req,
                             const data_user &user) const;

  dag_node_ptr build_node(std::unique_ptr<operation> op,
                          const requirements_list &requirements,
                          const execution_hints &hints);
  

  mutable std::mutex _mutex;
  dag _current_dag;
  runtime* _rt;
};


}
}

#endif