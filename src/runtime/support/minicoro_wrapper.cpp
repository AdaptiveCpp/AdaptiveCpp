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

/**
 * Defines MCO_ALLOC as malloc to avoid performance issues when the compiler
 * issues a malloc + memset for a calloc. The coroutines allocate more space
 * than they touch. Hence if the compiler issues a malloc + memset then all the
 * requested space written to and hence allocated in physical memory which is
 * slow. See AdaptiveCpp GitHub issue 1931. @edubart advises that this
 * substitution is safe.
 */
#include <stdlib.h>
#define MCO_ALLOC(size) malloc(size)
#define MCO_DEALLOC(ptr, size) free(ptr)
#define MINICORO_IMPL
#include "minicoro.h"

namespace hipsycl::rt::support {

fiber::fiber(function_type function, void* argument)
  : _coro(nullptr), _function(std::move(function)), _arg(argument)
{
  static constexpr size_t stack_size = 256 * 1024; // bytes
  create_coroutine(stack_size);
}

fiber::~fiber() {
  if (_coro) {
    mco_destroy(_coro);
    _coro = nullptr;
  }
}

yield_signal fiber::resume() {
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

void fiber::yield(yield_signal signal) {
  assert(_coro != nullptr);
  mco_push(_coro, &signal, sizeof(yield_signal));
  mco_yield(_coro);
}

fiber_status fiber::status() const {
  if (!_coro) return fiber_status::dead;

  switch (mco_status(_coro)) {
    case MCO_SUSPENDED: return fiber_status::suspended;
    case MCO_RUNNING:   return fiber_status::running;
    case MCO_DEAD:      return fiber_status::dead;
    default:            assert(false); return fiber_status::dead;
  }
}

bool fiber::is_alive() const {
  return status() != fiber_status::dead;
}

fiber* fiber::get_current() {
    mco_coro* co = mco_running();
    return static_cast<fiber*>(mco_get_user_data(co));
}

void fiber::create_coroutine(std::size_t stack_size) {
  mco_desc desc = mco_desc_init(entry_point, stack_size);
  desc.user_data = this;

  mco_result res = mco_create(&_coro, &desc);
  assert(res == MCO_SUCCESS);
}

void fiber::entry_point(mco_coro* co) {
  auto* self = static_cast<fiber*>(mco_get_user_data(co));
  self->_function(self);
}

} // namespace hipsycl::rt::support

