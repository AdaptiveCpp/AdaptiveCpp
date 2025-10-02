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

#include <cassert>

#include "hipSYCL/runtime/support/minicoro_wrapper.hpp"

#define MINICORO_IMPL
#include "minicoro.h"

namespace hipsycl::rt::support {
template<typename arguments_type>
fiber<arguments_type>::fiber(function_type func, const arguments_type& initial_args)
  : _coro(nullptr), _function(std::move(func)), _args(initial_args)
{
  static constexpr size_t stack_size = 256 * 1024; // bytes
  create_coroutine(stack_size);
}

template<typename arguments_type>
fiber<arguments_type>::~fiber() {
  if (_coro) {
    mco_destroy(_coro);
    _coro = nullptr;
  }
}

template<typename arguments_type>
yield_signal fiber<arguments_type>::resume() {
  assert(_coro != nullptr);
  assert(status() != fiber_status::dead);

  mco_result res = mco_resume(_coro);
  assert(res == MCO_SUCCESS);

  if (status() == fiber_status::dead)
    return yield_signal::dead;

  yield_signal signal = yield_signal::fail;
  std::size_t bytes = mco_get_bytes_stored(_coro);
  assert(bytes == sizeof(yield_signal));
  mco_pop(_coro, &signal, sizeof(yield_signal));
  return signal;
}

template<typename arguments_type>
void fiber<arguments_type>::yield(yield_signal signal) {
  assert(_coro != nullptr);
  mco_push(_coro, &signal, sizeof(yield_signal));
  mco_yield(_coro);
}

template<typename arguments_type>
arguments_type& fiber<arguments_type>::args() { return _args; }

template<typename arguments_type>
const arguments_type& fiber<arguments_type>::args() const { return _args; }

template<typename arguments_type>
fiber_status fiber<arguments_type>::status() const {
  if (!_coro) return fiber_status::dead;

  switch (mco_status(_coro)) {
    case MCO_SUSPENDED: return fiber_status::suspended;
    case MCO_RUNNING:   return fiber_status::running;
    case MCO_DEAD:      return fiber_status::dead;
    default:            assert(false); return fiber_status::dead;
  }
}

template<typename arguments_type>
bool fiber<arguments_type>::is_alive() const {
  return status() != fiber_status::dead;
}

template<typename arguments_type>
void fiber<arguments_type>::create_coroutine(std::size_t stack_size) {
  mco_desc desc = mco_desc_init(entry_point, stack_size);
  desc.user_data = this;

  mco_result res = mco_create(&_coro, &desc);
  assert(res == MCO_SUCCESS);
}

template<typename arguments_type>
void fiber<arguments_type>::entry_point(mco_coro* co) {
  auto* self = static_cast<fiber*>(mco_get_user_data(co));
  self->_function(self);
}
}

// Not beautiful, but helps keep it all isolated
#include "hipSYCL/glue/generic/host/collective_execution_engine.hpp"

namespace hipsycl::rt::support {
  // Expose the nested type for each Dim to allow explicit template instantiation
  template class fiber<hipsycl::glue::host::collective_execution_engine<1>::FiberData>;
  template class fiber<hipsycl::glue::host::collective_execution_engine<2>::FiberData>;
  template class fiber<hipsycl::glue::host::collective_execution_engine<3>::FiberData>;
}
