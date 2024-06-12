# `ACPP_EXT_MULTI_DEVICE_QUEUE`

This extension allows `sycl::queue` to automatically distribute work across multiple devices. The functionality from this extension requires that the scheduler type is set to `unbound` (default).

**Note:** This is highly experimental, not performance-optimized, the current work distribution algorithm is extremely primitive and a placeholder for a proper scheduling algorithm. This extension should not yet be used for any production workloads.

A multi-device queue can be constructed either by passing a vector of `sycl::device` to the queue constructor, or by using the new device selectors, such as `system_selector_v`. See the API reference below for details.

Additionally, the behavior of the default selector can be modified to behave like a system selector or a multi-gpu selector. See the documentation on environment variables for more details.

## Example

```c++
sycl::queue q{sycl::system_selector_v};
q.single_task([=](){
  // may be executed on any device inside the system
  // as automatically decided by the scheduler
}).wait();
```

## Restrictions

In multi-device mode, the following operations are currently unsupported, although support may be partially or fully added in the future:

* `queue::get_device()`;
* Overloads for USM memory management functions that are provided the queue as argument;
* the USM-related member functions of `sycl::handler` such as `memcpy`, `memset`, `fill`, `prefetch` and `prefetch_host`;
* If kernels use USM pointers, the user is responsible for making sure that the USM pointer is valid on all devices that the queue might dispatch to;
* All variants of `handler::copy()`;
* `handler::update()` from the `ACPP_EXT_UPDATE_DEVICE` extension

## API Reference

```c++
namespace sycl {

enum class selection_policy {
  all,
  best
};

// A wrapper class that turns a regular selector into a multi-device
// selector. 
// If selection policy is selection_policy::all, then
// all devices are selected for which the provided selector returns a
// non-negative number.
// If selection policy is selection_policy::best, then
// only devices with the best score are selected. Multiple devices are
// only selected in this case if the same top score is returned
// for multiple devices.
template <class Selector, selection_policy P = selection_policy::all>
class multi_device_selector {
public:
  constexpr multi_device_selector(const Selector& s = Selector{});

  int operator()(const device& dev);
};

inline constexpr multi_device_selector<cpu_selector,selection_policy::best>
    multi_cpu_selector_v;

// Returns all GPUs that have been targeted at compile time.
inline constexpr multi_device_selector<gpu_selector, selection_policy::best>
    multi_gpu_selector_v;

// Returns all devices in the system that have been targeted at
// compile time.
inline constexpr multi_device_selector<default_selector, selection_policy::all>
    system_selector_v;

class queue {
public:
  // New constructors that accept a vector of device.
  // Alternatively, an appropriate multi-device selector
  // can be provided to the queue.
  explicit queue(const std::vector<device> &devices,
                 const async_handler &handler,
                 const property_list &propList = {});

  explicit queue(const std::vector<device>& devices, 
                const property_list& propList = {});

  explicit queue(const context &syclContext,
                const std::vector<device> &devices,
                const property_list &propList = {});

  explicit queue(const context &syclContext,
                 const std::vector<device> &devices,
                 const async_handler &asyncHandler,
                 const property_list &propList = {});

  // Returns all devices that this queue might dispatch to
  std::vector<device> get_devices() const;
};

}
```