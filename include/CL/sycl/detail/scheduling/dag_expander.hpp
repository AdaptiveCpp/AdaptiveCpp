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

#ifndef HIPSYCL_DAG_EXPANDER_HPP
#define HIPSYCL_DAG_EXPANDER_HPP

#include "dag.hpp"
#include "dag_node.hpp"
#include "operations.hpp"
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

/// "Expands" a DAG by removing requirements or replacing
/// requirements with actual memory transfer operations.
/// The result is a DAG which can be executed by a backend executor.
///
/// Note: In order for the \c dag_expander to correctly decide
/// if memory transfers are required, the scheduler must already
/// have attached hints to all dag_nodes on which device operations
/// are scheduled to execute.
class dag_expander
{
public:
  dag_expander(dag* d);

  void undo_mark_nodes();

  void transform_dag();

private:

  void order_by_requirements(
    std::vector<dag_node_ptr>& ordered_nodes) const;

  std::unordered_map<dag_node_ptr, std::vector<dag_node_ptr>> _parents;

  dag* _dag;
};

}
}
}

#endif
