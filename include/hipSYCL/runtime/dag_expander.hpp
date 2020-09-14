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

#include "data.hpp"
#include "dag.hpp"
#include "dag_node.hpp"
#include "dag_enumerator.hpp"
#include "dag_scheduler.hpp"
#include "operations.hpp"
#include <vector>

namespace hipsycl {
namespace rt {


class dag_expander_annotation
{
public:

  dag_expander_annotation();

  void set_optimized_away();
  void add_replacement_operation(std::unique_ptr<operation> op);
  void set_forward_to_node(dag_node_ptr forward_to_node);

  bool is_optimized_away() const;
  bool is_operation_replaced() const;
  bool is_node_forwarded() const;

  const std::vector<std::unique_ptr<operation>> &
  get_replacement_operations() const;
  
  dag_node_ptr get_forwarding_target() const;

private:
  bool _optimized_away;
  std::vector<std::unique_ptr<operation>> _replacement_operations;
  dag_node_ptr _forwarding_target;
};


class dag_expansion_result
{
public:
  dag_expansion_result() = default;
  dag_expansion_result(const dag_enumerator& object_enumeration);

  void reset();

  dag_expander_annotation &node_annotations(std::size_t node_id);
  const dag_expander_annotation& node_annotations(std::size_t node_id) const;

  buffer_data_region *memory_state(std::size_t data_region_id);
  const buffer_data_region *memory_state(std::size_t data_region_id) const;

  buffer_data_region* original_data_region(std::size_t data_region_id);
  const buffer_data_region* original_data_region(std::size_t data_region_id) const;

  void add_data_region_fork(std::size_t data_region_id,
                            std::unique_ptr<buffer_data_region> fork,
                            buffer_data_region* original);
private:
  std::size_t _num_nodes;
  std::size_t _num_memory_buffers;
  std::vector<dag_expander_annotation> _node_annotations;
  std::vector<std::unique_ptr<buffer_data_region>> _forked_memory_states;
  std::vector<buffer_data_region*> _original_data_regions;
};



/// "Expands" a DAG by removing requirements or replacing
/// requirements with actual memory transfer operations.
/// The result is a DAG which can be executed by a backend executor.
///
/// Note: In order for the \c dag_expander to correctly decide
/// if memory transfers are required, the scheduler must already
/// have decided for all dag_nodes on which device operations
/// are scheduled to execute.
class dag_expander
{
public:
  /// Initializes the expander by performing preprocessing steps
  /// that can be reused for several expansions.
  dag_expander(const dag* d, const dag_enumerator& enumerator);

  void expand(const std::vector<node_scheduling_annotation> &sched_node_properties,
              dag_expansion_result& out) const;

private:
  dag_enumerator _enumerator;

  std::vector<dag_node_ptr> _ordered_nodes;

  void order_by_requirements(
    std::vector<dag_node_ptr>& ordered_nodes) const;

  const dag* _dag;
};

}
}

#endif
