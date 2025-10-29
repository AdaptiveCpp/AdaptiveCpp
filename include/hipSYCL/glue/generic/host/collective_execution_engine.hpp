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
#ifndef HIPSYCL_COLLECTIVE_EXECUTION_ENGINE_HPP
#define HIPSYCL_COLLECTIVE_EXECUTION_ENGINE_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

/**
 * Allow disabling fibers; and don't try using them in device pass.
 */
#if !defined(ACPP_NO_FIBERS) && !defined(SYCL_DEVICE_ONLY)
#define ACPP_USE_FIBERS
#endif

#ifdef ACPP_USE_FIBERS

#include <functional>
#include <vector>

#include "hipSYCL/runtime/support/minicoro_wrapper.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"

#include "iterate_range.hpp"
#include "range_decomposition.hpp"

namespace hipsycl {
namespace glue {
namespace host {

enum class group_execution_iteration {
  omp_for,
  sequential
};

template<int Dim>
class collective_execution_engine {
public:
  collective_execution_engine(
      sycl::range<Dim> num_groups, sycl::range<Dim> local_size,
      sycl::id<Dim> offset,
      const static_range_decomposition<Dim>& group_range_decomposition,
      int my_group_region)
      : _num_groups{num_groups}, _local_size{local_size}, _offset{offset},
        _fibers_spawned{false},
        _master_group_position(0), _groups{group_range_decomposition},
        _my_group_region{my_group_region} {}

  template <class WorkItemFunction>
  void run_kernel(WorkItemFunction f) {
    _kernel = f;
    _fibers_spawned = false;
    _master_group_position = 0;

    // Create master fiber
    auto& data = _fiber_args.emplace_back(FiberData{this, sycl::id<Dim>{}, 0});
    _fibers.emplace_back(master_coro_body, &data);
    bool all_done = false;

    // Launch master fiber
    if (yield_signal signal = _fibers[0].resume(); signal == yield_signal::dead) {
      all_done = true;
    } else {
      assert(signal == yield_signal::spawn);
      spawn_fibers();
    }

    while (!all_done) {
      all_done = true;
      for (auto& fiber : _fibers) {
        if (fiber.is_alive()) {
          if (yield_signal signal = fiber.resume(); signal != yield_signal::dead) {
            assert(signal == yield_signal::barrier || signal == yield_signal::next_item);
            all_done = false;
          }
        }
      }
    }

    // Cleanup
    _fiber_args.clear();
    _fibers.clear();
  }

  void barrier() {
    fiber* const self = fiber::get_current();
    if (!_fibers_spawned) {
      assert(self == &_fibers[0]);
      self->yield(yield_signal::spawn);
      assert(_fibers_spawned);
      self->yield(yield_signal::next_item);
    }
    self->yield(yield_signal::barrier);
  }

private:
  struct FiberData {
    collective_execution_engine* engine;
    sycl::id<Dim> local_id;
    size_t master_offset;
  };
  using fiber = hipsycl::rt::support::fiber;
  using yield_signal = hipsycl::rt::support::yield_signal;


  sycl::range<Dim> _num_groups;
  sycl::range<Dim> _local_size;
  sycl::id<Dim> _offset;
  bool _fibers_spawned;
  // Use deque to keep pointers stable on emplace_back
  std::deque<fiber> _fibers;
  std::deque<FiberData> _fiber_args;
  std::function<void(sycl::id<Dim>, sycl::id<Dim>)> _kernel;
  size_t _master_group_position;
  const static_range_decomposition<Dim>& _groups;
  int _my_group_region;

  static void master_coro_body(fiber* self) {
    collective_execution_engine* engine = self->arg<FiberData>()->engine;
    engine->_groups.for_each_local_element(
      engine->_my_group_region, [engine, self](sycl::id<Dim> group_id) {
        if (!engine->_fibers_spawned) {
          iterate_range(engine->_local_size, [&](sycl::id<Dim> local_id) {
            if (!engine->_fibers_spawned)
              engine->execute_work_item(local_id, group_id);
          });
        } else {
          self->yield(yield_signal::next_item);
          engine->execute_work_item(sycl::id<Dim>{}, group_id);
        }
        ++engine->_master_group_position;
      });
  }

  static void worker_coro_body(fiber* self) {
    FiberData* arg = self->arg<FiberData>();
    collective_execution_engine* engine = arg->engine;
    size_t current_group = 0;
    engine->_groups.for_each_local_element(
      engine->_my_group_region, [&](sycl::id<Dim> group_id) {
        if (current_group >= arg->master_offset) {
          self->yield(yield_signal::next_item);
          engine->execute_work_item(arg->local_id, group_id);
        }
        current_group++;
      });
  }

  void spawn_fibers() {
    bool first = true;
    iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
      if (!first) {
        auto& data = _fiber_args.emplace_back(FiberData{this, local_id, _master_group_position});
        _fibers.emplace_back(worker_coro_body, &data);
      }
      first = false;
    });
    _fibers_spawned = true;
  }

  void execute_work_item(sycl::id<Dim> local_id, sycl::id<Dim> group_id) {
    _kernel(local_id, group_id);
  }
};

}
}
} // namespace hipsycl

#endif // ACPP_USE_FIBERS

#endif
