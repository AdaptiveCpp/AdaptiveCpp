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
#ifndef HIPSYCL_SUPPORT_MINICORO_WRAPPER_HPP
#define HIPSYCL_SUPPORT_MINICORO_WRAPPER_HPP
#include <functional>

struct mco_coro;

namespace hipsycl::rt::support {

enum class fiber_status {
    suspended,
    running,
    dead
};

// Used only for debugging
enum class yield_signal {
    fail,
    dead,
    spawn,
    barrier,
    next_item
};

class fiber {
public:
    using function_type = std::function<void(fiber*)>;

    explicit fiber(function_type function, void* argument);

    fiber(const fiber&) = delete;
    fiber& operator=(const fiber&) = delete;
    fiber(fiber&& other) = delete;
    fiber& operator=(fiber&& other) = delete;
    ~fiber();

    yield_signal resume();
    void yield(yield_signal signal);

    template<typename T>
    T* arg() { return reinterpret_cast<T*>(_arg); }
    template<typename T>
    const T* arg() const { return reinterpret_cast<const T*>(_arg); }

    [[nodiscard]] bool is_alive() const;

    static fiber* get_current();

private:
    mco_coro* _coro;
    function_type _function;
    void* _arg;

    void create_coroutine(std::size_t stack_size);
    [[nodiscard]] fiber_status status() const;
    static void entry_point(mco_coro* co);
};

} // namespace hipsycl::rt::support

#endif // HIPSYCL_SUPPORT_MINICORO_WRAPPER_HPP
