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

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS


using all_ops = boost::mp11::mp_list<
    sycl::plus<>,
    sycl::multiplies<>,
    sycl::bit_and<>,
    sycl::bit_or<>,
    sycl::bit_xor<>,
    sycl::logical_and<>,
    sycl::logical_or<>,
    sycl::minimum<>,
    sycl::maximum<>
>;

BOOST_AUTO_TEST_CASE_TEMPLATE(group_known_identity, T, test_types) {
    
    boost::mp11::mp_for_each<all_ops>([&](auto op){
        static constexpr auto HAS_KNOWN_IDENTITY = sycl::has_known_identity_v<decltype(op), T>;
        if constexpr (HAS_KNOWN_IDENTITY) {
            static_assert(std::is_same_v<decltype(sycl::known_identity_v<decltype(op), T>), const T>);
        }
    });
}
#endif
