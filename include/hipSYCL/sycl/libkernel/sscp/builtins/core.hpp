/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_BUILTINS_CORE_HPP
#define HIPSYCL_SSCP_BUILTINS_CORE_HPP

#include "builtin_config.hpp"

#include <stddef.h>

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_x();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_y();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_z();

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_x();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_y();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_z();

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_x();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_y();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_z();

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_x();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_y();
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_z();

// This is used to implement the optimization in llvm-to-backend to treat
// all queries as fitting into int.
// The implementation is provided by the compiler and does not need to be implemented
// by backends.
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_if_global_sizes_fit_in_int();

template<int Dim, class T>
T __hipsycl_sscp_typed_get_global_linear_id() {
  if constexpr(Dim == 1) {
    T gid_x = (T)__hipsycl_sscp_get_group_id_x();
    T lsize_x = (T)__hipsycl_sscp_get_local_size_x();
    T lid_x = (T)__hipsycl_sscp_get_local_id_x();

    return gid_x * lsize_x + lid_x;
  } else if constexpr(Dim == 2) {
    T gid_x = (T)__hipsycl_sscp_get_group_id_x();
    T gid_y = (T)__hipsycl_sscp_get_group_id_y();
    T lsize_x = (T)__hipsycl_sscp_get_local_size_x();
    T lsize_y = (T)__hipsycl_sscp_get_local_size_y();
    T lid_x = (T)__hipsycl_sscp_get_local_id_x();
    T lid_y = (T)__hipsycl_sscp_get_local_id_y();
    T ngroups_x = (T)__hipsycl_sscp_get_num_groups_x();

    T id_x = gid_x * lsize_x + lid_x;
    T id_y = gid_y * lsize_y + lid_y;

    T global_size_x = lsize_x * ngroups_x;

    return global_size_x * id_y + id_x;
  } else if constexpr(Dim == 3) {
    T gid_x = (T)__hipsycl_sscp_get_group_id_x();
    T gid_y = (T)__hipsycl_sscp_get_group_id_y();
    T gid_z = (T)__hipsycl_sscp_get_group_id_z();
    T lsize_x = (T)__hipsycl_sscp_get_local_size_x();
    T lsize_y = (T)__hipsycl_sscp_get_local_size_y();
    T lsize_z = (T)__hipsycl_sscp_get_local_size_z();
    T lid_x = (T)__hipsycl_sscp_get_local_id_x();
    T lid_y = (T)__hipsycl_sscp_get_local_id_y();
    T lid_z = (T)__hipsycl_sscp_get_local_id_z();
    T ngroups_x = (T)__hipsycl_sscp_get_num_groups_x();
    T ngroups_y = (T)__hipsycl_sscp_get_num_groups_y();
    
    T id_x = gid_x * lsize_x + lid_x;
    T id_y = gid_y * lsize_y + lid_y;
    T id_z = gid_z * lsize_z + lid_z;

    T global_size_x = lsize_x * ngroups_x;
    T global_size_y = lsize_y * ngroups_y;

    return global_size_x * global_size_y * id_z + global_size_x * id_y + id_x;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __hipsycl_sscp_typed_get_local_linear_id() {
  if constexpr(Dim == 1) {
    return (T)__hipsycl_sscp_get_local_id_x();
  } else if constexpr(Dim == 2) {
    T lid_x = (T)__hipsycl_sscp_get_local_id_x();
    T lid_y = (T)__hipsycl_sscp_get_local_id_y();

    T lsize_x = (T)__hipsycl_sscp_get_local_size_x();

    return lsize_x * lid_y + lid_x;
  } else if constexpr(Dim == 3) {
    T lid_x = (T)__hipsycl_sscp_get_local_id_x();
    T lid_y = (T)__hipsycl_sscp_get_local_id_y();
    T lid_z = (T)__hipsycl_sscp_get_local_id_z();

    T lsize_x = (T)__hipsycl_sscp_get_local_size_x();
    T lsize_y = (T)__hipsycl_sscp_get_local_size_y();

    return lsize_x * lsize_y * lid_z + lsize_x * lid_y + lid_x;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __hipsycl_sscp_typed_get_group_linear_id() {
  if constexpr(Dim == 1) {
    return (T)__hipsycl_sscp_get_group_id_x();
  } else if constexpr(Dim == 2) {
    T gid_x = (T)__hipsycl_sscp_get_group_id_x();
    T gid_y = (T)__hipsycl_sscp_get_group_id_y();

    T ngroups_x = (T)__hipsycl_sscp_get_num_groups_x();

    return ngroups_x * gid_y + gid_x;
  } else if constexpr(Dim == 3) {
    T gid_x = (T)__hipsycl_sscp_get_group_id_x();
    T gid_y = (T)__hipsycl_sscp_get_group_id_y();
    T gid_z = (T)__hipsycl_sscp_get_group_id_z();

    T ngroups_x = (T)__hipsycl_sscp_get_num_groups_x();
    T ngroups_y = (T)__hipsycl_sscp_get_num_groups_y();

    return ngroups_x * ngroups_y * gid_z + ngroups_x * gid_y + gid_x;
  } else {
    return 0;
  }
}

template<int Dim, class T>
T __hipsycl_sscp_typed_get_global_size() {
  if constexpr(Dim == 1) {
    return (T)__hipsycl_sscp_get_local_size_x() * (T)__hipsycl_sscp_get_num_groups_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__hipsycl_sscp_get_local_size_x() * (T)__hipsycl_sscp_get_num_groups_x();
    T size_y = (T)__hipsycl_sscp_get_local_size_y() * (T)__hipsycl_sscp_get_num_groups_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__hipsycl_sscp_get_local_size_x() * (T)__hipsycl_sscp_get_num_groups_x();
    T size_y = (T)__hipsycl_sscp_get_local_size_y() * (T)__hipsycl_sscp_get_num_groups_y();
    T size_z = (T)__hipsycl_sscp_get_local_size_z() * (T)__hipsycl_sscp_get_num_groups_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __hipsycl_sscp_typed_get_local_size() {
  if constexpr(Dim == 1) {
    return (T)__hipsycl_sscp_get_local_size_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__hipsycl_sscp_get_local_size_x();
    T size_y = (T)__hipsycl_sscp_get_local_size_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__hipsycl_sscp_get_local_size_x();
    T size_y = (T)__hipsycl_sscp_get_local_size_y();
    T size_z = (T)__hipsycl_sscp_get_local_size_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __hipsycl_sscp_typed_get_num_groups() {
  if constexpr(Dim == 1) {
    return (T)__hipsycl_sscp_get_num_groups_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__hipsycl_sscp_get_num_groups_x();
    T size_y = (T)__hipsycl_sscp_get_num_groups_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__hipsycl_sscp_get_num_groups_x();
    T size_y = (T)__hipsycl_sscp_get_num_groups_y();
    T size_z = (T)__hipsycl_sscp_get_num_groups_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}



template<int Dim>
size_t __hipsycl_sscp_get_global_linear_id() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_global_linear_id<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_global_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __hipsycl_sscp_get_group_linear_id() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_group_linear_id<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_group_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __hipsycl_sscp_get_local_linear_id() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_local_linear_id<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_local_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __hipsycl_sscp_get_global_size() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_global_size<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_global_size<Dim, size_t>();
  }
}

template<int Dim>
size_t __hipsycl_sscp_get_local_size() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_local_size<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_local_size<Dim, size_t>();
  }
}

template<int Dim>
size_t __hipsycl_sscp_get_num_groups() {
  if(__hipsycl_sscp_if_global_sizes_fit_in_int()) {
    return __hipsycl_sscp_typed_get_num_groups<Dim, int>();
  } else {
    return __hipsycl_sscp_typed_get_num_groups<Dim, size_t>();
  }
}

#endif
