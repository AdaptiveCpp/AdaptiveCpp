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
#ifndef HIPSYCL_DAG_MULTI_EVENT_HPP
#define HIPSYCL_DAG_MULTI_EVENT_HPP

#include <vector>
#include <memory>
#include "../event.hpp"

namespace hipsycl {
namespace rt {

class dag_multi_node_event : public dag_node_event
{
public:
  dag_multi_node_event(std::vector<std::shared_ptr<dag_node_event>> events)
  : _events(events)
  {}

  virtual bool is_complete() const override {
    for(auto& evt : _events)
      if(!evt->is_complete())
        return false;
    return true;
  }

  virtual void wait() override {
    for(auto& evt : _events)
      evt->wait();
  }

  virtual ~dag_multi_node_event() {}

  void add_event(std::shared_ptr<dag_node_event> evt){
    _events.push_back(evt);
  }

private:
  std::vector<std::shared_ptr<dag_node_event>> _events;
};

}
}

#endif