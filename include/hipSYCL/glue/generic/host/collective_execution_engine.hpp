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
 * Due to an issue with Boost.Intrusive (used by Boost.Fiber), on Windows,
 * which is triggered in device pass of Clang CUDA, we may only use this in host pass.
 * This should not be a problem, as this implementation is anyways just required during host pass.
 */
#if !defined(HIPSYCL_NO_FIBERS) && !defined(SYCL_DEVICE_ONLY)
#define HIPSYCL_HAS_FIBERS
#endif

#ifdef HIPSYCL_HAS_FIBERS

#include <functional>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/barrier.hpp>

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
      const static_range_decomposition<Dim> &group_range_decomposition,
      int my_group_region)
      : _num_groups{num_groups}, _local_size{local_size}, _offset{offset},
        _group_barrier{local_size.size()}, _fibers_spawned{false},
        _fibers(local_size.size()), _groups{group_range_decomposition},
        _my_group_region{my_group_region} {}

  template <class WorkItemFunction>
  void run_kernel(WorkItemFunction f) {
    _kernel = f;
    _fibers_spawned = false;
    _master_group_position = 0;

    // Try sequential processing (using only one fiber) - if
    // other fibers need to be spawned, only process first work item
    // as other work items will be processed by other fibers
    _fibers[0] = boost::fibers::fiber([this]() {
      _groups.for_each_local_element(
          _my_group_region, [this](sycl::id<Dim> group_id) {
            if (!_fibers_spawned) {
              iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
                if (!_fibers_spawned)
                  execute_work_item(local_id, group_id);
              });
            } else {
              barrier();
              // Only execute work item 0 from now on
              execute_work_item(sycl::id<Dim>{}, group_id);
            }
            ++_master_group_position;
          });
    });

    if (_fibers.size() > 0) {
      _fibers[0].join();
      if(_fibers_spawned){
        for (std::size_t i = 1; i < _fibers.size(); ++i) {
          _fibers[i].join();
        }
      }
    }

  }

  void barrier() {
    if(!_fibers_spawned){
      // We are still in sequential processing mode,
      // need to spawn the other fibers
      spawn_fibers();
      // Perform additional barrier on master fiber
      // to participate in the other fibers initial barrier
      // when entering the first group
      barrier();
    }
    
    _group_barrier.wait();
  }

private:
  // Spawn remaining fibers
  void spawn_fibers() {

    std::size_t n = 0;
    iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
      // First work item will be processed by master fiber
      if (n != 0) {
        
        std::size_t master_offset = _master_group_position;
        _fibers[n] = boost::fibers::fiber([local_id, this, master_offset]() {
          std::size_t current_group = 0;
          _groups.for_each_local_element(
              _my_group_region, [&, this](sycl::id<Dim> group_id) {
                if (current_group >= master_offset) {
                  barrier();
                  execute_work_item(local_id, group_id);
                }
                ++current_group;
              });
        });
      }
      ++n;
    });

    _fibers_spawned = true;
  }

  void execute_work_item(sycl::id<Dim> local_id, sycl::id<Dim> group_id) {
    _kernel(local_id, group_id);
  }

  sycl::range<Dim> _num_groups;
  sycl::range<Dim> _local_size;
  sycl::id<Dim> _offset;
  boost::fibers::barrier _group_barrier;
  bool _fibers_spawned;
  std::vector<boost::fibers::fiber> _fibers;
  std::function<void(sycl::id<Dim>, sycl::id<Dim>)> _kernel;
  std::size_t _master_group_position;
  const static_range_decomposition<Dim> &_groups;
  int _my_group_region;
};

}
}
} // namespace hipsycl

#endif // HIPSYCL_HAS_FIBERS

#endif
