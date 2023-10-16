/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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


#include <algorithm>

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/allocator.hpp"

namespace hipsycl {
namespace rt {

namespace {

void abort_submission(dag_node_ptr node) {
  for (auto weak_req : node->get_requirements()) {
    if(auto req = weak_req.lock()) {
      if (!req->is_submitted()) {
        req->cancel();
      }
    }
  }
  node->cancel();
}

template <class Handler>
void execute_if_buffer_requirement(dag_node_ptr node, Handler h) {
  if (node->get_operation()->is_requirement()) {
    if (cast<requirement>(node->get_operation())->is_memory_requirement()) {
      if (cast<memory_requirement>(node->get_operation())
              ->is_buffer_requirement()) {
        h(cast<buffer_memory_requirement>(node->get_operation()));
      }
    }
  }
}

void assign_devices_or_default(dag_node_ptr node, device_id default_device) {
  if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
    node->assign_to_device(default_device);
  } else {
    node->assign_to_device(node->get_execution_hints()
                               .get_hint<hints::bind_to_device>()
                               ->get_device_id());
  }
}

// Initialize memory accesses for requirements
void initialize_memory_access(buffer_memory_requirement *bmem_req,
                              device_id target_dev) {
  assert(bmem_req);

  void *device_pointer = bmem_req->get_data_region()->get_memory(target_dev);
  bmem_req->initialize_device_data(device_pointer);
  HIPSYCL_DEBUG_INFO << "dag_direct_scheduler: Setting device data pointer of "
                        "requirement node " << dump(bmem_req) << " to " 
                     << device_pointer << std::endl;
}

result ensure_allocation_exists(runtime *rt,
                                buffer_memory_requirement *bmem_req,
                                device_id target_dev) {
  assert(bmem_req);
  if (!bmem_req->get_data_region()->has_allocation(target_dev)) {
    const std::size_t num_bytes =
        bmem_req->get_data_region()->get_num_elements().size() *
        bmem_req->get_data_region()->get_element_size();

    backend_allocator *allocator =
        rt->backends().get(target_dev.get_backend())->get_allocator(target_dev);
    // Currently we just pass 0 for the alignment which should
    // cause backends to align to the largest supported type.
    // TODO: A better solution might be to select a custom alignment
    // best on sizeof(T). This requires querying backend alignment capabilities.
    void *ptr = allocator->allocate(0, num_bytes);

    if(!ptr)
      return register_error(
                 __hipsycl_here(),
                 error_info{
                     "dag_direct_scheduler: Lazy memory allocation has failed.",
                     error_type::memory_allocation_error});

    bmem_req->get_data_region()->add_empty_allocation(target_dev, ptr,
                                                      allocator);
  }

  return make_success();
}

void for_each_explicit_operation(
    dag_node_ptr node, std::function<void(operation *)> explicit_op_handler) {
  if (node->is_submitted())
    return;
  
  if (!node->get_operation()->is_requirement()) {
    explicit_op_handler(node->get_operation());
    return;
  } else {
    execute_if_buffer_requirement(node,
                                  [&](buffer_memory_requirement *bmem_req) {
          
          device_id target_device = node->get_assigned_device();

          std::vector<range_store::rect> outdated_regions;
          bmem_req->get_data_region()->get_outdated_regions(
              target_device, bmem_req->get_access_offset3d(),
              bmem_req->get_access_range3d(), outdated_regions);

          for (range_store::rect region : outdated_regions) {
            std::vector<std::pair<device_id, range_store::rect>> update_sources;

            bmem_req->get_data_region()->get_update_source_candidates(
                target_device, region, update_sources);

            if (update_sources.empty()) {
              register_error(
                  __hipsycl_here(),
                  error_info{"dag_direct_scheduler: Could not obtain data "
                             "update sources when trying to materialize "
                             "implicit requirement"});
              node->cancel();
              return;
            }

            // Just use first source for now:
            memory_location src{update_sources[0].first,
                                update_sources[0].second.first,
                                bmem_req->get_data_region()};
            memory_location dest{target_device, region.first,
                                 bmem_req->get_data_region()};
            std::unique_ptr<operation> op =
                std::make_unique<memcpy_operation>(src, dest, region.second);

            explicit_op_handler(op.get());
            /// TODO This has to be changed once we support multi-operation nodes
            node->assign_effective_operation(std::move(op));
          }
        });
  }
}

std::pair<backend_executor *, device_id>
select_executor(runtime *rt, dag_node_ptr node, operation *op) {
  device_id dev = node->get_assigned_device();

  assert(!op->is_requirement());

  // If we have been requested to run on a particular executor, do this.
  backend_executor* user_preferred_executor = nullptr;
  if(node->get_execution_hints().has_hint<hints::prefer_executor>()){
    user_preferred_executor= node->get_execution_hints()
        .get_hint<hints::prefer_executor>()
        ->get_executor();
  }

  backend_id executor_backend; device_id preferred_device;
  if (op->has_preferred_backend(executor_backend, preferred_device)) {
    // If we want an executor from a different backend, we may need to pass
    // a different device id.

    if (user_preferred_executor &&
        user_preferred_executor->can_execute_on_device(preferred_device))
        return std::make_pair(user_preferred_executor, preferred_device);

    return std::make_pair(
        rt->backends().get(executor_backend)->get_executor(preferred_device),
        preferred_device);

  } else {
    
    if (user_preferred_executor &&
        user_preferred_executor->can_execute_on_device(dev))
      return std::make_pair(user_preferred_executor, dev);

    return std::make_pair(
        rt->backends().get(dev.get_backend())->get_executor(dev), dev);
  }
}

void submit(backend_executor *executor, dag_node_ptr node, operation *op) {
  
  node_list_t reqs;
  node->for_each_nonvirtual_requirement([&](dag_node_ptr req) {
    if(std::find(reqs.begin(), reqs.end(), req) == reqs.end())
      reqs.push_back(req);
  });
  // Compress requirements by removing complete requirements
  reqs.erase(std::remove_if(
                 reqs.begin(), reqs.end(),
                 [](dag_node_ptr elem) { return elem->is_known_complete(); }),
             reqs.end());

  node->assign_to_executor(executor);
  
  executor->submit_directly(node, op, reqs);
  assert(node->is_submitted());
  // After node submission, no additional instrumentations can be added.
  // Marking as complete causes code that waits for instrumentation results
  // to proceed to waiting on the requested instrumentation.
  op->get_instrumentations().mark_set_complete();
}

result submit_requirement(runtime* rt, dag_node_ptr req) {
  if (!req->get_operation()->is_requirement() || req->is_submitted())
    return make_success();

  sycl::access::mode access_mode = sycl::access::mode::read_write;

  // Make sure that all required allocations exist
  // (they must exist when we try initialize device pointers!)
  result res = make_success();
  execute_if_buffer_requirement(req, [&](buffer_memory_requirement *bmem_req) {
    res = ensure_allocation_exists(rt, bmem_req, req->get_assigned_device());
    access_mode = bmem_req->get_access_mode();
  });
  if (!res.is_success())
    return res;
  
  // Then initialize memory accesses
  execute_if_buffer_requirement(
    req, [&](buffer_memory_requirement *bmem_req) {
      initialize_memory_access(bmem_req, req->get_assigned_device());
  });

  // Don't create memcopies if access is discard
  if (access_mode != sycl::access::mode::discard_write &&
      access_mode != sycl::access::mode::discard_read_write) {
    bool has_initialized_content = true;
    execute_if_buffer_requirement(
        req, [&](buffer_memory_requirement *bmem_req) {
          has_initialized_content =
              bmem_req->get_data_region()->has_initialized_content(
                  bmem_req->get_access_offset3d(),
                  bmem_req->get_access_range3d());
        });
    if(has_initialized_content){
      for_each_explicit_operation(req, [&](operation *op) {
        if (!op->is_data_transfer()) {
          res = make_error(
              __hipsycl_here(),
              error_info{
                  "dag_direct_scheduler: only data transfers are supported "
                  "as operations generated from implicit requirements.",
                  error_type::feature_not_supported});
        } else {
          std::pair<backend_executor *, device_id> execution_config =
              select_executor(rt, req, op);
          // TODO What if we need to copy between two device backends through
          // host?
          
          // TODO: The following is super-hacky and hints that we might
          // have to do some larger architectural changes here:
          //
          // For host accessors, their target device will be set to the host
          // device, but that might not be the correct device to carry
          // out the memcpy, because the host device cannot access e.g. GPU memory.
          // So we set the assigned device to the one we get from select_executor()
          // which takes such considerations into account.
          // Without this, since the executor tries to execute on the device
          // assigned to the node, it might attempt to dispatch to some invalid device.
          //
          // However, at the end of submit() below this section, we then try to
          // to update the data state for device assigned to the node.
          // For a host accessor, because we have in fact updated the host memory,
          // get_assigned_device() must return the original device at this point.
          //
          // Currently we solve this by retrieving the original assigned device,
          // changing it to the one we wish to carry out the memcpy during submission,
          // and then change back afterwards.
          // This is not pretty. Is this a hint that there are actual two different
          // parts of the DAG node that we should distinguish architecturally - the SYCL
          // view and the backend execution view?
          auto original_device = req->get_assigned_device();
          req->assign_to_device(execution_config.second);
          submit(execution_config.first, req, op);
          req->assign_to_device(original_device);
        }
      });
    } else {
      HIPSYCL_DEBUG_WARNING
          << "dag_direct_scheduler: Detected a requirement that is neither of "
             "discard access mode (SYCL 1.2.1) nor no_init property (SYCL 2020) "
             "that accesses uninitialized data. Consider changing to "
             "discard/no_init. Optimizing potential data transfers away."
          << std::endl;
    }
  }
  if (!res.is_success())
    return res;

  // If the requirement did not result in any operations...
  if (!req->get_event()) {
    // create dummy event
    req->mark_virtually_submitted();
  }
  // This must be executed even if the requirement did
  // not result in actual operations in order to make sure
  // that regions are valid after discard accesses 
  execute_if_buffer_requirement(
      req, [&](buffer_memory_requirement *bmem_req) {
        if (access_mode == sycl::access::mode::read) {
          bmem_req->get_data_region()->mark_range_valid(
              req->get_assigned_device(), bmem_req->get_access_offset3d(),
              bmem_req->get_access_range3d());
        } else {
          bmem_req->get_data_region()->mark_range_current(
              req->get_assigned_device(), bmem_req->get_access_offset3d(),
              bmem_req->get_access_range3d());
        }
      });

  
  return make_success();
}
}

dag_direct_scheduler::dag_direct_scheduler(runtime* rt)
: _rt{rt} {}

void dag_direct_scheduler::submit(dag_node_ptr node) {
  if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
    register_error(__hipsycl_here(),
                   error_info{"dag_direct_scheduler: Direct scheduler does not "
                              "support DAG nodes not bound to devices.",
                              error_type::feature_not_supported});
    abort_submission(node);
    return;
  }

  device_id target_device = node->get_execution_hints()
                                .get_hint<hints::bind_to_device>()
                                ->get_device_id();
  node->assign_to_device(target_device);
  
  for (auto weak_req : node->get_requirements()) {
    if(auto req = weak_req.lock())
      assign_devices_or_default(req, target_device);
  }

  for (auto weak_req : node->get_requirements()) {
    if(auto req = weak_req.lock()) {
      if (!req->get_operation()->is_requirement()) {
        if (!req->is_submitted()) {
          register_error(__hipsycl_here(),
                    error_info{"dag_direct_scheduler: Direct scheduler does not "
                                "support processing multiple unsubmitted nodes",
                                error_type::feature_not_supported});
          abort_submission(node);
          return;
        }
      } else {
        result res = submit_requirement(_rt, req);

        if (!res.is_success()) {
          register_error(res);
          abort_submission(node);
          return;
        }
      }
    }
  }

  if (node->get_operation()->is_requirement()) {
    result res = submit_requirement(_rt, node);
    
    if (!res.is_success()) {
      register_error(res);
      abort_submission(node);
      return;
    }
  } else {
    // TODO What if this is an explicit copy between two device backends through
    // host?
    std::pair<backend_executor *, device_id> execution_config =
        select_executor(_rt, node, node->get_operation());
    // If we are not a requirement, execution target should not have changed
    // -- only e.g. for host accessors a change is expected as they target the
    // CPU device, but their memcyp may need to be executed on a device.
    assert(execution_config.second == target_device);
    rt::submit(execution_config.first, node, node->get_operation());
  }
}

}
}
