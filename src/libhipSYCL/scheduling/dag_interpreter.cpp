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


#include <cassert>

#include "CL/sycl/detail/scheduling/dag_interpreter.hpp"
#include "CL/sycl/detail/scheduling/hints.hpp"

namespace cl {
namespace sycl {
namespace detail {

dag_interpreter::dag_interpreter(const dag* d, const dag_enumerator *enumerator,
                                 const dag_expansion_result *expansion_result)
    : _expansion{expansion_result} {}

bool dag_interpreter::is_node_real(const dag_node_ptr &node) const
{
  std::size_t node_id = this->get_node_id(node);

  return !_expansion->node_annotations(node_id).is_optimized_away();
}

std::size_t dag_interpreter::get_node_id(const dag_node_ptr &node) const
{
  assert(node->get_execution_hints().has_hint(
      execution_hint_type::dag_enumeration_id));

  return node->get_execution_hints()
      .get_hint<hints::dag_enumeration_id>()
      ->id();
}

}
}
}