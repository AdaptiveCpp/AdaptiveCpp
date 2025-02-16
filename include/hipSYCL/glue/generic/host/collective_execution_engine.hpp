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

#include "minicoro.h"

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

namespace yield_kind {
// Some odd value not easily confused with pointers
static void* spawn = reinterpret_cast<void*>(0xff07);
static void* barrier = reinterpret_cast<void*>(0xff09);
static void* next_item = reinterpret_cast<void*>(0xff11);
}
static constexpr size_t fiber_stack_size = 256*1024;

template<int Dim>
class collective_execution_engine {
public:
  collective_execution_engine(
      sycl::range<Dim> num_groups, sycl::range<Dim> local_size,
      sycl::id<Dim> offset,
      const static_range_decomposition<Dim>& group_range_decomposition,
      int my_group_region)
      : _num_groups{num_groups}, _local_size{local_size}, _offset{offset},
        _fibers_spawned{false}, _fibers(local_size.size(), nullptr),
        _groups{group_range_decomposition}, _my_group_region{my_group_region},
        _current_coro{nullptr} {}

  template <class WorkItemFunction>
  void run_kernel(WorkItemFunction f) {
    _kernel = f;
    _fibers_spawned = false;
    _master_group_position = 0;

    // Create master coroutine
    mco_desc desc = mco_desc_init(master_entry, fiber_stack_size);
    desc.user_data = this;
    mco_coro* master_co;
    mco_result res = mco_create(&master_co, &desc);
    assert(res == MCO_SUCCESS);
    _fibers[0] = master_co;

    bool all_done = false;

    // Launch master coroutine
    void* result = resume(_fibers[0]);
    if (mco_status(_fibers[0]) == MCO_DEAD) {
      all_done = true;
    } else {
      assert(result == yield_kind::spawn);
      spawn_fibers();
    }

    while (!all_done) {
      all_done = true;
      void* master_yield_kind = nullptr;
      for (auto& co : _fibers) {
        if (co && mco_status(co) != MCO_DEAD) {
          void* result = resume(co);
          if (mco_status(co) != MCO_DEAD) {
            assert(result == yield_kind::barrier || result == yield_kind::next_item);
            if (!master_yield_kind) master_yield_kind = result;
            assert(result == master_yield_kind && "Inconsistent yield reasons");
            all_done = false;
          }
        }
      }
    }

    // Cleanup
    for (auto& co : _fibers) {
      if (co) {
        mco_destroy(co);
        co = nullptr;
      }
    }
  }

  void* resume(mco_coro* co) {
    _current_coro = co;
    mco_result res = mco_resume(co);
    _current_coro = nullptr;

    if (res != MCO_SUCCESS) {
      assert(false && "mco_resume failed");
      return nullptr;
    }

    if (mco_status(co) == MCO_DEAD)
      return nullptr;

    void* yield_kind = nullptr;
    size_t bytes = mco_get_bytes_stored(co);
    if (bytes >= sizeof(void*))
      mco_pop(co, &yield_kind, sizeof(void*));
    return yield_kind;
  }

  void barrier() {
    assert(_current_coro && "Barrier outside coroutine");
    if (!_fibers_spawned) {
      mco_push(_current_coro, &yield_kind::spawn, sizeof(void*));
      mco_yield(_current_coro);
      mco_push(_current_coro, &yield_kind::next_item, sizeof(void*));
      mco_yield(_current_coro);
    }
    mco_push(_current_coro, &yield_kind::barrier, sizeof(void*));
    mco_yield(_current_coro);
  }

private:
  sycl::range<Dim> _num_groups;
  sycl::range<Dim> _local_size;
  sycl::id<Dim> _offset;
  bool _fibers_spawned;
  std::vector<mco_coro*> _fibers;
  std::function<void(sycl::id<Dim>, sycl::id<Dim>)> _kernel;
  size_t _master_group_position;
  const static_range_decomposition<Dim>& _groups;
  int _my_group_region;
  mco_coro* _current_coro;

  static void master_entry(mco_coro* co) {
    auto* engine = static_cast<collective_execution_engine*>(mco_get_user_data(co));
    engine->master_coro_body(co);
  }

  void master_coro_body(mco_coro* co) {
    _groups.for_each_local_element(
      _my_group_region, [this, co](sycl::id<Dim> group_id) {
        if (!_fibers_spawned) {
          iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
            if (!_fibers_spawned)
              execute_work_item(local_id, group_id);
          });
        } else {
          assert(co == _current_coro);
          mco_push(co, &yield_kind::next_item, sizeof(void*));
          mco_yield(co);
          execute_work_item(sycl::id<Dim>{}, group_id);
        }
        ++_master_group_position;
      });
  }

  struct CoroutineData {
    collective_execution_engine* engine;
    sycl::id<Dim> local_id;
    size_t master_offset;
  };

  static void worker_entry(mco_coro* co) {
    auto* data = static_cast<CoroutineData*>(mco_get_user_data(co));
    data->engine->worker_coro_body(co, data->local_id, data->master_offset);
    delete data;
  }

  void worker_coro_body(mco_coro* co, sycl::id<Dim> local_id, size_t master_offset) {
    size_t current_group = 0;
    _groups.for_each_local_element(
      _my_group_region, [&](sycl::id<Dim> group_id) {
        if (current_group >= master_offset) {
          assert(co == _current_coro);
          mco_push(co, &yield_kind::next_item, sizeof(void*));
          mco_yield(co);
          execute_work_item(local_id, group_id);
        }
        current_group++;
      });
  }

  void spawn_fibers() {
    size_t n = 0;
    iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
      if (n != 0) {
        auto* data = new CoroutineData{this, local_id, _master_group_position};
        mco_desc desc = mco_desc_init(worker_entry, fiber_stack_size);
        desc.user_data = data;
        mco_coro* worker_co;
        mco_result res = mco_create(&worker_co, &desc);
        assert(res == MCO_SUCCESS);
        _fibers[n] = worker_co;
      }
      n++;
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
