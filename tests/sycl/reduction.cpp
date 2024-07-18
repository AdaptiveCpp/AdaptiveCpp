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

#include <numeric>
#include <algorithm>
#include <type_traits>

#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(reduction_tests, reset_device_fixture)


auto tolerance = boost::test_tools::tolerance(0.001f);

template <class T, class Generator, class Handler, class BinaryOp>
void test_scalar_reduction(sycl::queue &q, const T& identity,
                           std::size_t num_elements,
                           BinaryOp op, Generator gen, Handler h) {
  T* test_data = sycl::malloc_shared<T>(num_elements, q);
  T* output_data = sycl::malloc_shared<T>(1, q);

  for(std::size_t i = 0; i < num_elements;++i)
    test_data[i] = gen(i);
  
  h(test_data, output_data);
  q.wait();

  T expected_result =
      std::accumulate(test_data, test_data + num_elements, identity, op);
  
  if constexpr(std::is_floating_point_v<T>) {
    BOOST_TEST(expected_result == *output_data, tolerance);
  } else {
    BOOST_TEST(expected_result == *output_data);
  }

  sycl::free(test_data, q);
  sycl::free(output_data, q);
}

template<class T, class BinaryOp>
struct input_generator {};

template<class T>
struct input_generator <T, sycl::plus<T>> {
  T operator() (std::size_t i) const {
    return static_cast<T>(i);
  }
};

template<class T>
struct input_generator <T, sycl::multiplies<T>> {
  T operator() (std::size_t i) const {
    if(i > 1500)
      return static_cast<T>(1);

    if(i % 100 == 0)
      return static_cast<T>(2);
    
    return static_cast<T>(1);
  }
};

template<class T, class BinaryOp, int Line>
class reduction_kernel;

template<class T, class BinaryOp>
void test_single_reduction(std::size_t input_size, std::size_t local_size, 
                          const T& identity, BinaryOp op){
  sycl::queue q;

  auto input_gen = [&](std::size_t i){
    return input_generator<T,BinaryOp>{}(i);
  };

  test_scalar_reduction(
      q, identity, input_size, op, input_gen, [&](T *input, T *output) {
        q.parallel_for<reduction_kernel<T, BinaryOp, __LINE__>>(
            sycl::range<1>(input_size),
            sycl::reduction(
                output, identity, op,
                sycl::property_list{
                    sycl::property::reduction::initialize_to_identity{}}),
            [=](sycl::id<1> idx, auto &reducer) {
              reducer.combine(input[idx[0]]);
            });
      });

  if(input_size % local_size == 0) {
    test_scalar_reduction(
        q, identity, input_size, op, input_gen, [&](T *input, T *output) {
          q.parallel_for<reduction_kernel<T, BinaryOp, __LINE__>>(
              sycl::nd_range(sycl::range{input_size}, sycl::range{local_size}),
              sycl::reduction(
                  output, identity, op,
                  sycl::property_list{
                      sycl::property::reduction::initialize_to_identity{}}),
              [=](sycl::nd_item<1> idx, auto &reducer) {
                reducer.combine(input[idx.get_global_linear_id()]);
              });
        });
  }
}


template<class T>
void test_two_reductions(std::size_t input_size, std::size_t local_size){
  sycl::queue q;

  std::size_t num_reductions = 2;

  std::vector<T*> inputs, outputs;
  for(std::size_t i = 0; i < num_reductions; ++i) {
    inputs.push_back(sycl::malloc_shared<T>(input_size, q));
    outputs.push_back(sycl::malloc_shared<T>(1, q));
  }

  for(std::size_t reduction = 0; reduction < num_reductions; ++reduction) {
    for(std::size_t i = 0; i < input_size; ++i) {
      if(reduction == 0)
        inputs[reduction][i] = input_generator<T, sycl::plus<T>>{}(i);
      else
        inputs[reduction][i] = input_generator<T, sycl::multiplies<T>>{}(i);
    }
  }

  T* input0 = inputs[0];
  T* input1 = inputs[1];
  T* output0 = outputs[0];
  T* output1 = outputs[1];

  auto verify = [&]() {
    q.wait();

    T expected_add_result =
      std::accumulate(input0, input0 + input_size, T{0}, std::plus<T>{});

    T expected_mul_result =
      std::accumulate(input1, input1 + input_size, T{1}, std::multiplies<T>{});

    if constexpr(std::is_floating_point_v<T>) {
      BOOST_TEST(expected_add_result == *output0, tolerance);
      BOOST_TEST(expected_mul_result == *output1, tolerance);
    } else {
      BOOST_TEST(expected_add_result == *output0);
      BOOST_TEST(expected_mul_result == *output1);
    }
    *output0 = T{};
    *output1 = T{};
  };

  q.parallel_for<reduction_kernel<T, class MultiOp, __LINE__>>(
      sycl::range<1>(input_size),
      sycl::reduction(output0, T{0}, sycl::plus<T>{},
                      sycl::property_list{
                          sycl::property::reduction::initialize_to_identity{}}),
      sycl::reduction(output1, T{1}, sycl::multiplies<T>{},
                      sycl::property_list{
                          sycl::property::reduction::initialize_to_identity{}}),
      [=](sycl::id<1> idx, auto &add_reducer, auto &mul_reducer) {
        add_reducer += input0[idx[0]];
        mul_reducer *= input1[idx[0]];
      });

  verify();

  if(input_size % local_size == 0) {

    q.parallel_for<reduction_kernel<T, class MultiOp, __LINE__>>(
        sycl::nd_range(sycl::range{input_size}, sycl::range{local_size}),
        sycl::reduction(
            output0, T{0}, sycl::plus<T>{},
            sycl::property_list{
                sycl::property::reduction::initialize_to_identity{}}),
        sycl::reduction(
            output1, T{1}, sycl::multiplies<T>{},
            sycl::property_list{
                sycl::property::reduction::initialize_to_identity{}}),
        [=](sycl::nd_item<1> idx, auto &add_reducer, auto &mul_reducer) {
          add_reducer += input0[idx.get_global_linear_id()];
#ifdef __ACPP_USE_ACCELERATED_CPU__
          idx.barrier(); // workaround for omp simd failure as below.
#endif
          mul_reducer *= input1[idx.get_global_linear_id()];
        });

    verify();
  }

  q.wait();
  for(std::size_t i = 0; i < num_reductions; ++i) {
    if(inputs[i])
      sycl::free(inputs[i], q);
    if(outputs[i])
      sycl::free(outputs[i], q);
  }
  
}


using all_test_types = boost::mpl::list<char, unsigned int, int, long long, float, 
  double>;

using large_test_types = boost::mpl::list<unsigned int, int, long long, float, 
  double>;

using very_large_test_types = boost::mpl::list<unsigned int, long long, float, 
  double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(single_kernel_single_scalar_reduction, T, all_test_types) {
  test_single_reduction(128, 128, T{0}, sycl::plus<T>{});
  test_single_reduction(128, 128, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(two_kernels_single_scalar_reduction, T, large_test_types) {
  test_single_reduction(128*128, 128, T{0}, sycl::plus<T>{});
  test_single_reduction(128*128, 128, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(three_kernels_single_scalar_reduction, T, very_large_test_types) {
  test_single_reduction(64*64*64, 64, T{0}, sycl::plus<T>{});
  test_single_reduction(64*64*64, 64, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mismatching_group_size, T, large_test_types) {
  test_single_reduction(1000, 128, T{0}, sycl::plus<T>{});
  test_single_reduction(1000, 128, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(oversized_group_size, T, all_test_types) {
  test_single_reduction(20, 128, T{0}, sycl::plus<T>{});
  test_single_reduction(20, 128, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(two_reductions, T, large_test_types) {
  test_two_reductions<T>(128*128, 128);
}

BOOST_AUTO_TEST_CASE(accessor_reduction) {
  sycl::queue q;
  sycl::buffer<int> values_buff{1024};
  { 
    sycl::host_accessor a{values_buff};
    std::iota(a.get_pointer(), a.get_pointer() + 1024, 0);
  }
  
  sycl::buffer<int> sum_buff{1};
  sycl::buffer<int> max_buff{1};

  q.submit([&](sycl::handler &cgh) {
    auto values_acc = values_buff.get_access<sycl::access_mode::read>(
        cgh);

    sycl::accessor sum_acc {sum_buff, cgh};
    sycl::accessor max_acc {max_buff, cgh};

    auto sumReduction = sycl::reduction(
        sum_acc, sycl::plus<int>(),
        sycl::property_list{
            sycl::property::reduction::initialize_to_identity{}});
    auto maxReduction = sycl::reduction(
        max_acc, sycl::maximum<int>(),
        sycl::property_list{
            sycl::property::reduction::initialize_to_identity{}});

    cgh.parallel_for(sycl::range<1>{1024}, sumReduction, maxReduction,
                     [=](sycl::id<1> idx, auto &sum, auto &max) {
                       sum += values_acc[idx];
                       max.combine(values_acc[idx]);
                     });
  });
  
  BOOST_CHECK(max_buff.get_host_access()[0] == 1023);
  BOOST_CHECK(sum_buff.get_host_access()[0] == 523776);
}


BOOST_AUTO_TEST_CASE(buffer_reduction) {
  sycl::queue q;
  sycl::buffer<int> values_buff{1024};
  { 
    sycl::host_accessor a{values_buff};
    std::iota(a.get_pointer(), a.get_pointer() + 1024, 0);
  }
  
  sycl::buffer<int> sum_buff{1};
  sycl::buffer<int> max_buff{1};

  q.submit([&](sycl::handler &cgh) {
    auto values_acc = values_buff.get_access<sycl::access_mode::read>(
        cgh);

    auto sumReduction = sycl::reduction(
        sum_buff, cgh, sycl::plus<int>(),
        sycl::property_list{
            sycl::property::reduction::initialize_to_identity{}});
    auto maxReduction = sycl::reduction(
        max_buff, cgh, sycl::maximum<int>(),
        sycl::property_list{
            sycl::property::reduction::initialize_to_identity{}});

    cgh.parallel_for(sycl::range<1>{1024}, sumReduction, maxReduction,
                     [=](sycl::id<1> idx, auto &sum, auto &max) {
                       sum += values_acc[idx];
                       max.combine(values_acc[idx]);
                     });
  });
  
  BOOST_CHECK(max_buff.get_host_access()[0] == 1023);
  BOOST_CHECK(sum_buff.get_host_access()[0] == 523776);
}

BOOST_AUTO_TEST_CASE(incremental_reduction) {
  const int size = 1024;
  sycl::queue q;
  int* data = sycl::malloc_shared<int>(size, q);

  int* result = sycl::malloc_shared<int>(1, q);
  for(std::size_t i = 0; i < size;++i)
    data[i] = static_cast<int>(i);

  *result = 0;

  // Also tests plus<> without explicit template argument
  q.parallel_for(size, sycl::reduction(result, sycl::plus<>()),
                 [=](auto idx, auto &redu) { redu += data[idx]; }).wait();

  int expected_result = std::accumulate(data, data + size, 0);
  BOOST_CHECK(*result == expected_result);

  q.parallel_for(size, sycl::reduction(result, sycl::plus<>()),
                 [=](auto idx, auto &redu) { redu += data[idx]; }).wait();

  BOOST_CHECK(*result == 2 * expected_result);

  sycl::free(data, q);
  sycl::free(result, q);
}

BOOST_AUTO_TEST_CASE(chain_combine_reductions) {
const int size = 1024;
  sycl::queue q;
  int* data = sycl::malloc_shared<int>(size, q);

  int* result = sycl::malloc_shared<int>(1, q);
  for(std::size_t i = 0; i < size;++i)
    data[i] = static_cast<int>(i);

  int expected_result = 2 * std::accumulate(data, data + size, 0);

  *result = 0;
  q.parallel_for(size, sycl::reduction(result, sycl::plus<>()),
                 [=](auto idx, auto &redu) {
		   (redu += data[idx]) += data[idx];
		 }).wait();
  BOOST_CHECK(*result == expected_result);

  *result = 0;
  q.parallel_for(size, sycl::reduction(result, sycl::plus<>()),
                 [=](auto idx, auto &redu) {
		   redu.combine(data[idx]).combine(data[idx]);
		 }).wait();

  BOOST_CHECK(*result == expected_result);

  sycl::free(data, q);
  sycl::free(result, q);
}

BOOST_AUTO_TEST_SUITE_END()
