#ifndef SYCU_EVENT_HPP
#define SYCU_EVENT_HPP

#include "types.hpp"
#include "backend/backend.hpp"
#include "exception.hpp"
#include "info/event.hpp"

namespace cl {
namespace sycl {

namespace detail {

class event_manager
{
public:
  event_manager()
  {
    detail::check_error(hipEventCreate(&_evt));
  }

  ~event_manager()
  {
    detail::check_error(hipEventDestroy(_evt));
  }

  hipEvent_t& get_event()
  {
    return _evt;
  }
private:
  hipEvent_t _evt;
};

using event_ptr = shared_ptr_class<event_manager>;

} // detail

class event {

public:
  event()
    : _ready{true}
  {}

  event(const detail::event_ptr& evt)
    : _ready{false}, _evt{evt}
  {}

  /* CL Interop is not supported
  event(cl_event clEvent, const context& syclContext);

  cl_event get();
  */

  vector_class<event> get_wait_list()
  {
    return vector_class<event>{};
  }

  void wait()
  {
    this->wait_until_done();
  }

  static void wait(const vector_class<event> &eventList)
  {
    for(const event& evt: eventList)
      evt.wait_until_done();
  }

  void wait_and_throw()
  {
    wait();
  }

  static void wait_and_throw(const vector_class<event> &eventList)
  {
    wait(eventList);
  }

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const
  { throw unimplemented{"event::get_info() is unimplemented."}; }

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type get_profiling_info() const
  { throw unimplemented{"event::get_profiling_info() is unimplemented."}; }

  bool operator ==(const event& rhs) const
  { return _evt == rhs._evt; }

  bool operator !=(const event& rhs) const
  { return !(*this == rhs); }
private:
  void wait_until_done() const
  {
    if(!_ready)
      detail::check_error(hipEventSynchronize(_evt->get_event()));
  }

  bool _ready;
  detail::event_ptr _evt;
};

namespace detail {
/// Inserts an event into the current stream
static event insert_event(hipStream_t stream)
{
  event_ptr evt{new event_manager()};
  detail::check_error(hipEventRecord(evt->get_event(), stream));
  return event{evt};
}

} // detail

} // namespace sycl
} // namespace cl

#endif
