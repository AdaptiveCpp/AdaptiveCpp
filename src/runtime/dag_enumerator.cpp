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
#include <unordered_set>
#include "hipSYCL/runtime/dag_enumerator.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/sycl/detail/debug.hpp"

namespace hipsycl {
namespace rt {

dag_enumerator::dag_enumerator(dag *d)
    : _num_nodes{0}, _num_data_regions{0}
{
  this->enumerate_nodes(d);
  this->enumerate_data_regions(d);
}

std::size_t dag_enumerator::get_node_index_space_size() const
{
  return _num_nodes;
}

std::size_t dag_enumerator::get_data_region_index_space_size() const
{
  return _num_data_regions;
}


void dag_enumerator::enumerate_nodes(dag *d)
{
  this->_num_nodes = 0;

  d->for_each_node([this](dag_node_ptr node) {
    assert(!node->has_node_id());

    node->assign_node_id(this->_num_nodes);
    
    this->_num_nodes++;
  });
}

void dag_enumerator::enumerate_data_regions(dag *d) {
  this->_num_data_regions = 0;

  std::unordered_set<buffer_data_region*> processed_data_regions;

  for (dag_node_ptr mem_req : d->get_memory_requirements()) {
    assert_is<memory_requirement>(mem_req->get_operation());

    if (cast<memory_requirement>(mem_req->get_operation())
            ->is_buffer_requirement()) {

      auto *data = cast<buffer_memory_requirement>(mem_req->get_operation())
                       ->get_data_region()
                       .get();

      if (processed_data_regions.find(data) == processed_data_regions.end()) {
        
        if (data->has_id()) {
          HIPSYCL_DEBUG_WARNING << "dag_enumerator: Setting id on data region "
                                << data << " that already has id." << std::endl;
        }

        data->set_enumerated_id(_num_data_regions);
        processed_data_regions.insert(data);

        ++_num_data_regions;
      } else {
        assert(data->has_id());
      }

    }
  }
}


}
}
