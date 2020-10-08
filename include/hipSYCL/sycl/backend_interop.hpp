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


#include "hipSYCL/glue/backend_interop.hpp"

#include "backend.hpp"
#include "access.hpp"
#include "platform.hpp"
#include "device.hpp"
#include "context.hpp"
#include "buffer.hpp"
#include "kernel.hpp"
#include "event.hpp"
#include "libkernel/accessor.hpp"
#include "libkernel/stream.hpp"

#ifndef HIPSYCL_SYCL_BACKEND_INTEROP_HPP
#define HIPSYCL_SYCL_BACKEND_INTEROP_HPP

namespace hipsycl {
namespace sycl {

class queue;

namespace detail {


template <class T>
struct interop_traits {};

#define HIPSYCL_DEFINE_INTEROP_TRAIT(sycl_type, interop_trait_type)            \
  template <> struct interop_traits<sycl_type> {                               \
    template <backend B>                                                       \
    using native_type = typename glue::backend_interop<B>::interop_trait_type; \
  };

HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::device, native_device_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::platform, native_platform_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::context, native_context_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::queue, native_queue_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::event, native_event_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::stream, native_stream_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::kernel, native_kernel_type)
HIPSYCL_DEFINE_INTEROP_TRAIT(sycl::device_event, native_device_event_type)

template <typename dataT, int dimensions, access::mode accessmode,
          access::target Target, access::placeholder isPlaceholder>
struct interop_traits<
    sycl::accessor<dataT, dimensions, accessmode, Target, isPlaceholder>> {

  template <backend B>
  using native_type = typename glue::backend_interop<B>::native_mem_type;
};

} // namespace detail


template <backend Backend> class backend_traits {
public:
  template <class T>
  using native_type =
      typename detail::interop_traits<T>::template native_type<Backend>;

  using errc = typename glue::backend_interop<Backend>::error_type;
};

template <backend Backend>
typename backend_traits<Backend>::template native_type<device>
get_native(const device &sycl_object) {
  return glue::backend_interop<Backend>::get_native_device(sycl_object);
}

template <backend Backend>
typename backend_traits<Backend>::template native_type<platform>
get_native(const platform &sycl_object) {
  return glue::backend_interop<Backend>::get_native_platform(sycl_object);
}

template <backend Backend>
typename backend_traits<Backend>::template native_type<context>
get_native(const context &sycl_object) {
  return glue::backend_interop<Backend>::get_native_context(sycl_object);
}

template <backend Backend>
typename backend_traits<Backend>::template native_type<queue>
get_native(const queue &sycl_object) {
  return glue::backend_interop<Backend>::get_native_queue(sycl_object);
}

template <backend Backend>
typename backend_traits<Backend>::template native_type<event>
get_native(const event &sycl_object) {
  return glue::backend_interop<Backend>::get_native_event(sycl_object);
}

template <backend Backend>
typename backend_traits<Backend>::template native_type<buffer>
get_native(const event &sycl_object) {
  return glue::backend_interop<Backend>::get_native_event(sycl_object);
}

template <backend Backend>
platform make_platform(
    const typename backend_traits<Backend>::template native_type<platform>
        &backend_object) {
  return glue::backend_interop<Backend>::make_sycl_platform(backend_object); 
}

template <backend Backend>
device
make_device(const typename backend_traits<Backend>::template native_type<device>
                &backend_object) {
  return glue::backend_interop<Backend>::make_sycl_device(backend_object);
}

template <backend Backend>
context make_context(
    const typename backend_traits<Backend>::template native_type<context>
        &backend_object,
    const async_handler handler = {}) {

  return glue::backend_interop<Backend>::make_sycl_context(backend_object,
                                                           handler);
}

/*
We don't support make_queue() interop as it's antithetical to the way
queues work in hipSYCL, since there's no relation between a sycl::queue
and a backend object.

template <backend Backend>
queue make_queue(
    const typename backend_traits<Backend>::template native_type<queue>
        &backend_object,
    const context &ctx, const async_handler handler = {}) {

  return glue::backend_interop<Backend>::make_sycl_queue(backend_object, ctx,
                                                         handler);
}
*/

template <backend Backend>
event make_event(
    const typename backend_traits<Backend>::template native_type<event>
        &backend_object,
    const context &ctx) {

  return glue::backend_interop<Backend>::make_sycl_event(backend_object, ctx);
}

template <backend Backend>
buffer<int, 1> // TODO: How can infer the template arguments of buffer?
make_buffer(const typename backend_traits<Backend>::template native_type<buffer>
                &backend_object,
            const context &ctx, event available_event = {}) {
  return glue::backend_interop<Backend>::make_sycl_buffer(backend_object, ctx,
                                                          available_event);
}

/* -- We don't have image types yet in hipSYCL

template <backend Backend>
sampled_image make_sampled_image(
const backend_traits<Backend>::native_type<sampled_image> &backendObject,
const context &targetContext, image_sampler imageSampler, event availableEvent =
{});

template<backend Backend> unsampled_image make_unsampled_image( const
backend_traits<Backend>::native_type<unsampled_image> &backendObject, const
context &targetContext, event availableEvent = {});

template<backend Backend>
image_sampler make_image_sampler(
const backend_traits<Backend>::native_type<image_sampler> &backendObject,
const context &targetContext);
*/

template <backend Backend>
stream
make_stream(const typename backend_traits<Backend>::template native_type<stream>
                &backend_object,
            const context &ctx, event available_event = {}) {

  return glue::backend_interop<Backend>::make_sycl_stream(backend_object, ctx,
                                                          available_event);
}

template <backend Backend>
kernel
make_kernel(const typename backend_traits<Backend>::template native_type<kernel>
                &backend_object,
            const context &ctx) {
  return glue::backend_interop<Backend>::make_sycl_kernel(backend_object, ctx);
}

template <backend Backend>
kernel
make_module(const typename backend_traits<Backend>::template native_type<event>
                &backend_object,
            const context &ctx) {
  return glue::backend_interop<Backend>::make_sycl_module(backend_object, ctx);
}

}
}

#endif
