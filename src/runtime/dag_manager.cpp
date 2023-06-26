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

#include <memory>
#include <mutex>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/dag_unbound_scheduler.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/runtime.hpp"

namespace hipsycl {
namespace rt {


dag_build_guard::~dag_build_guard()
{
  _mgr->trigger_flush_opportunity();
}

dag_manager::dag_manager(runtime *rt)
    : _builder{std::make_unique<dag_builder>(rt)},
      _direct_scheduler{rt}, _unbound_scheduler{rt}, _rt{rt} {
  HIPSYCL_DEBUG_INFO << "dag_manager: DAG manager is alive!" << std::endl;
}

dag_manager::~dag_manager()
{
  HIPSYCL_DEBUG_INFO << "dag_manager: Waiting for async worker..." << std::endl;
  
  flush_sync();
  wait();

  HIPSYCL_DEBUG_INFO << "dag_manager: Shutdown." << std::endl;
}

dag_builder* 
dag_manager::builder() const
{
  return _builder.get();
}

void dag_manager::flush_async()
{
  HIPSYCL_DEBUG_INFO << "dag_manager: Submitting asynchronous flush..."
                     << std::endl;
  // This lock ensures that the submission process has atomic semantics.
  // In particular, it is important that once we have popped the latest
  // nodes from the DAG builder using finish_and_reset(), we directly submit them
  // to the worker thread.
  // Otherwise, the order in which submissions are processed in the worker thread
  // can be incorrect. This can cause queue::submit();flush_sync() to fail in
  // actually ensuring submission, or introduce dependencies in nodes during submission
  //  to other nodes that have not yet been submitted.
  std::lock_guard<std::mutex> lock{_flush_mutex};

  if(_builder->get_current_dag_size() > 0){
    dag new_dag = _builder->finish_and_reset();

    if(new_dag.num_nodes() > 0) {
      _worker([this, new_dag](){
        HIPSYCL_DEBUG_INFO << "dag_manager [async]: Flushing!" << std::endl;
        
        for(dag_node_ptr req : new_dag.get_memory_requirements()){
          assert_is<memory_requirement>(req->get_operation());

          memory_requirement *mreq =
              cast<memory_requirement>(req->get_operation());

          if(mreq->is_buffer_requirement()) {
            
            HIPSYCL_DEBUG_INFO
                << "dag_manager [async]: Releasing dead users of data region "
                << cast<buffer_memory_requirement>(mreq)->get_data_region().get()
                << std::endl;

            cast<buffer_memory_requirement>(mreq)
                ->get_data_region()
                ->get_users()
                .release_dead_users();
          }
          else
            assert(false && "Non-buffer requirements are unsupported");
        }

        // Go!!!
        scheduler_type stype =
            application::get_settings().get<setting::scheduler_type>();
        
        // This is okay because get_command_groups() returns
        // the nodes in the order they were submitted. This
        // makes it safe to submit them in this order to the direct scheduler.
        for(auto node : new_dag.get_command_groups()){
          HIPSYCL_DEBUG_INFO
                << "dag_manager [async]: Submitting node to scheduler!"
                << std::endl;
          if(stype == scheduler_type::direct) {
            _direct_scheduler.submit(node);
          } else if(stype == scheduler_type::unbound) {
            _unbound_scheduler.submit(node);
          }
        }
        HIPSYCL_DEBUG_INFO << "dag_manager [async]: DAG flush complete."
                          << std::endl;

        // Register nodes as submitted with the runtime
        for(auto node : new_dag.get_command_groups())
          this->register_submitted_ops(node);
        for(auto node : new_dag.get_memory_requirements())
          this->register_submitted_ops(node);

        if (this->_submitted_ops.get_num_nodes() >
            application::get_settings().get<setting::gc_trigger_batch_size>())
          this->_submitted_ops.async_wait_and_unregister();
      });
    }
  } else {
    HIPSYCL_DEBUG_INFO << "dag_manager: Nothing to do" << std::endl;
  }
}

void dag_manager::flush_sync()
{
  this->flush_async();
  // In a flush_sync, we can assume that we have finished a submission burst.
  // So this may be a good time to clean up and perform garbage collection!
  this->_submitted_ops.async_wait_and_unregister();
  
  HIPSYCL_DEBUG_INFO << "dag_manager: waiting for async worker..."
                     << std::endl;
  _worker.wait();
}

void dag_manager::wait()
{
  this->_submitted_ops.wait_for_all();
}

void dag_manager::wait(std::size_t node_group_id) {
  this->_submitted_ops.wait_for_group(node_group_id);
}

void dag_manager::register_submitted_ops(dag_node_ptr node) {
  this->_submitted_ops.update_with_submission(node);
}

void dag_manager::trigger_flush_opportunity()
{
  HIPSYCL_DEBUG_INFO << "dag_manager: Checking DAG flush opportunity..."
                     << std::endl;

  if (application::get_settings().get<setting::scheduler_type>() ==
      scheduler_type::direct) {
    // Direct scheduler always needs flushing
    flush_async();
  } else {
    if (builder()->get_current_dag_size() >
        application::get_settings().get<setting::max_cached_nodes>())
      flush_async();
  }
}

std::vector<dag_node_ptr> dag_manager::get_group(std::size_t node_group_id) {
  return _submitted_ops.get_group(node_group_id);
}
}
}
