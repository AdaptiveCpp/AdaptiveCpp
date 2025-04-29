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

#ifndef HIPSYCL_COMMON_CONFIG_HPP
#define HIPSYCL_COMMON_CONFIG_HPP

#define ACPP_VERSION_MAJOR 25
#define ACPP_VERSION_MINOR 02
#define ACPP_VERSION_PATCH 0
#define ACPP_VERSION_SUFFIX "+git.c38fc3af.20250708.branch.develop.dirty"

#define HIPSYCL_VERSION_MAJOR ACPP_VERSION_MAJOR
#define HIPSYCL_VERSION_MINOR ACPP_VERSION_MINOR
#define HIPSYCL_VERSION_PATCH ACPP_VERSION_PATCH

#define HIPSYCL_INSTALL_PREFIX "/home/jakob/projects/AdaptiveCpp/installation"
#define HIPSYCL_RT_LIBRARY_NAME "libacpp-rt.so"
#define HIPSYCL_COMMON_LIBRARY_NAME ""

// Macros to switch between proper std::filesystem, experimental version found in GCC 7, or Boost::filesystem
#define HIPSYCL_CXX_FILESYSTEM_HEADER <filesystem>
#define HIPSYCL_CXX_FILESYSTEM_NAMESPACE std::filesystem

#endif

