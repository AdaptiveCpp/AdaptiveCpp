/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_STREAM_HPP
#define HIPSYCL_STREAM_HPP

#include "../backend/backend.hpp"
#include "../types.hpp"
#include "../device.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

class stream_manager;
using stream_ptr = shared_ptr_class<stream_manager>;

class stream_manager
{
public:
  /// Creates a stream manager object on the default stream.
  /// On the default stream, there can be several stream manager objects.
  stream_manager();

  /// Creates a stream manager object on the default stream
  /// \param handler The error handler
  stream_manager(async_handler handler);

  /// Creates a new stream on the given device.
  /// \param d The device
  stream_manager(const device& d);

  /// Creates a new stream on the given device.
  /// \param d The device
  /// \param handler The error handler
  stream_manager(const device& d,
                 async_handler handler);

  /// If the managed stream is not the default stream,
  /// synchronizes and destroys the stream.
  ~stream_manager();

  /// \return The managed stream
  hipStream_t get_stream() const;

  /// \return A stream manager pointer using the default stream
  static stream_ptr default_stream();

  /// Sets the device on which the stream is constructed
  /// as active device
  void activate_device() const;

  /// \return The error handler associated with this
  /// stream
  async_handler get_error_handler() const;

  /// \return The device to which this stream is bound
  device get_device() const;
private:
  hipStream_t _stream;

  device _dev;
  async_handler _handler;
};


}
}
}


#endif
