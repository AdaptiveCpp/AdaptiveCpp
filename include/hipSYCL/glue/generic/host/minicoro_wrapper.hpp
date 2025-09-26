#ifndef HIPSYCL_COROUTINE_WRAPPER_HPP
#define HIPSYCL_COROUTINE_WRAPPER_HPP

#include <functional>

struct mco_coro;

namespace hipsycl {
namespace glue {
namespace host {

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

template<typename arguments_type = void*>
class fiber {
public:
    using function_type = std::function<void(fiber*)>;

    explicit fiber(function_type func, const arguments_type& initial_args);

    fiber(const fiber&) = delete;
    fiber& operator=(const fiber&) = delete;
    fiber(fiber&& other) = delete;
    fiber& operator=(fiber&& other) = delete;
    ~fiber();

    yield_signal resume();
    void yield(yield_signal signal);

    arguments_type& args();
    const arguments_type& args() const;

    [[nodiscard]] bool is_alive() const;

private:
    mco_coro* _coro;
    function_type _function;
    arguments_type _args;

    void create_coroutine(std::size_t stack_size);
    [[nodiscard]] fiber_status status() const;
    static void entry_point(mco_coro* co);
};

} // namespace host
} // namespace glue
} // namespace hipsycl

#endif // HIPSYCL_COROUTINE_WRAPPER_HPP
