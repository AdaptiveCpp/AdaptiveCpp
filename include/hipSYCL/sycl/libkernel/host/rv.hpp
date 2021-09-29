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

#ifndef HIPSYCL_RV_HPP
#define HIPSYCL_RV_HPP
#ifdef HIPSYCL_HAS_RV
#include <cstdint>

extern "C" bool rv_any(bool);
extern "C" bool rv_all(bool);
extern "C" std::uint32_t rv_ballot(bool);
extern "C" std::uint32_t rv_popcount(bool);
extern "C" std::uint32_t rv_index(bool);
extern "C" std::uint32_t rv_mask();
// rv_compact(float V, bool M)
extern "C" std::uint32_t rv_lane_id();
extern "C" std::uint32_t rv_num_lanes();
extern "C" float rv_extract(float, std::uint32_t);
extern "C" float rv_insert(float, std::uint32_t, float);
//extern "C" float rv_store(float*);
extern "C" float rv_shuffle(float, std::int32_t);
//extern "C" void rv_align(void*, std::int32_t);
#endif
#endif // HIPSYCL_RV_HPP
