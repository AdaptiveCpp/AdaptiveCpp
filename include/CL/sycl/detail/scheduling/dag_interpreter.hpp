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

#ifndef HIPSYCL_DAG_INTERPRETER_HPP
#define HIPSYCL_DAG_INTERPRETER_HPP

#include "hints.hpp"
#include "dag_expander.hpp"
#include "dag_enumerator.hpp"


namespace cl {
namespace sycl {
namespace detail {

class dag_interpreter
{
public:
  dag_interpreter(const dag* d, const dag_enumerator* enumerator,
                  const dag_expansion_result *expansion_result);

  /// After dag_expansion, a node may effectively depend on more requirements
  /// than originally specified. This function iterates over all effective
  /// requirements, calling a handler function for each requirement.
  /// \param h The handler function which must have the signature
  /// \c void(dag_node_ptr).
  template<class Handler>
  void for_each_requirement(const dag_node_ptr& node, Handler h) const
  {
    std::size_t node_id = get_node_id(node);

    const dag_expander_annotation &node_annotation =
        _expansion->node_annotations(node_id);

    if(node_annotation.is_optimized_away())
      return;

    if(node_annotation.is_node_forwarded()) {
      // Forwarded nodes must take into account their own requirements
      // as well as the requirements of the forwarding targets
      for(dag_node_ptr req : node->get_requirements())
        h(req);
      
      for_each_requirement(node_annotation.get_forwarding_target(), h);
    }
    else {
      for(dag_node_ptr req : node->get_requirements())
        h(req);
    }
    // TODO Third case: Node that is forwarded to
    // TODO What about the situation when requirements are already submitted?
  }

  /// After dag_expansion, a node may be optimized away or may have been
  /// translated into multiple operations. This function iterates over all
  /// operations relevant for this node.
  /// \param h The handler function called for each operation with the signature
  /// \c void(operation*)
  template<class Handler>
  void for_each_operation(const dag_node_ptr& node, Handler h) const
  {
    std::size_t node_id = get_node_id(node);

    const dag_expander_annotation &node_annotation =
        _expansion->node_annotations(node_id);
    if(node_annotation.is_optimized_away())
      return;
    else if(node_annotation.is_operation_replaced()){
      const auto &replacement_ops =
          node_annotation.get_replacement_operations();

      for(const auto& op : replacement_ops) {
        h(op.get());
      }
    }
    else if(node_annotation.is_node_forwarded()) {
      // TODO This may incorrectly lead to a double treatment of the node
      // since client code has no means of detecting that different nodes
      // point to the same operation
      for_each_operation(node_annotation.get_forwarding_target(), h);
    }
    else
      h(node->get_operation());
  }

  bool is_node_real(const dag_node_ptr& node) const;
private:
  std::size_t get_node_id(const dag_node_ptr& node) const;

  const dag_expansion_result* _expansion;
};

}
}
}

#endif
