/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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


#ifndef HIPSYCL_BIT_CAST_HPP
#define HIPSYCL_BIT_CAST_HPP

namespace hipsycl {
namespace sycl {

template <class Tout, class Tin>
Tout bit_cast(Tin x) {
  static_assert(sizeof(Tout)==sizeof(Tin), "Types must match sizes");

#if __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(Tout, x);
#elif __has_builtin(__builtin_memcpy_inline)
  Tout out;
  __builtin_memcpy_inline(&out, &x, sizeof(Tin));
  return out;
#else
  Tout out;
  __hipsycl_if_target_host(memcpy(&out, &x, sizeof(Tin)););
  __hipsycl_if_target_device(
    char* cout = reinterpret_cast<char*>(&out);
    char* cin =  reinterpret_cast<char*>(&x);
    for(int i = 0; i < sizeof(Tin); ++i)
      cout[i] = cin[i];
  );
  return out;
#endif
}


}
}

#endif
