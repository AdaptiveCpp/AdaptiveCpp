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
  dag_builder();

  dag_node_ptr add_kernel(std::unique_ptr<operation> op,
                          const requirements_list& requirements,
                          const execution_hints& hints = {});
  dag_node_ptr add_memcpy(std::unique_ptr<operation> op,
                          const requirements_list& requirements,
                          const execution_hints& hints = {});
  dag_node_ptr add_fill(std::unique_ptr<operation> op,
                        const requirements_list& requirements,
                        const execution_hints& hints = {});
  dag_node_ptr add_prefetch(std::unique_ptr<operation> op,
                            const requirements_list &requirements,
                            const execution_hints &hints = {});
  dag_node_ptr add_memset(std::unique_ptr<operation> op,
                          const requirements_list &requirements,
                          const execution_hints &hints = {});
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
  dag_node_ptr add_command_group(std::unique_ptr<operation> op,
                                const requirements_list& requirements,
                                const execution_hints& hints = {});

  mutable std::mutex _mutex;
  dag _current_dag;
};


}
}

#endif