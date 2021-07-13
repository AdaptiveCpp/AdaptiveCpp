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

#include <numeric>
#include <algorithm>
#include <type_traits>

#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(reduction_tests, reset_device_fixture)

auto tolerance = boost::test_tools::tolerance(0.001f);

template <class T, class Generator, class Handler, class BinaryOp>
void test_scalar_reduction_with_properties(sycl::queue &q, std::size_t num_elements,
    BinaryOp op, Generator gen, Handler h, const sycl::property_list &prop_list = {}) {
  T* test_data = sycl::malloc_shared<T>(num_elements, q);
  T* output_data = sycl::malloc_shared<T>(1, q);
  const T initial_output_value = 42;

  for(std::size_t i = 0; i < num_elements;++i)
    test_data[i] = gen(i);
  output_data[0] = initial_output_value;

  h(test_data, output_data, prop_list);
  q.wait();

  T accumulate_initial = sycl::known_identity_v<BinaryOp, T>;
  if (!prop_list.has_property<sycl::property::reduction::initialize_to_identity>()) {
    accumulate_initial = initial_output_value;
  }
  T expected_result =
      std::accumulate(test_data, test_data + num_elements, accumulate_initial, op);

  if constexpr(std::is_floating_point_v<T>) {
    BOOST_TEST(expected_result == *output_data, tolerance);
  } else {
    BOOST_TEST(expected_result == *output_data);
  }

  sycl::free(test_data, q);
  sycl::free(output_data, q);
}

template <class T, class Generator, class Handler, class BinaryOp>
void test_scalar_reduction(sycl::queue &q, std::size_t num_elements,
                           BinaryOp op, Generator gen, Handler h) {
  // test_scalar_reduction_with_properties<T>(q, num_elements, op, gen, h);
  test_scalar_reduction_with_properties<T>(q, num_elements, op, gen, h,
      sycl::property::reduction::initialize_to_identity{});
}

template<class T, class BinaryOp>
struct input_generator {};

template<class T, typename U>
struct input_generator <T, sycl::plus<U>> {
  T operator() (std::size_t i) const {
    return static_cast<T>(i);
  }
};

template<class T, typename U>
struct input_generator <T, sycl::multiplies<U>> {
  T operator() (std::size_t i) const {
    if(i > 1500)
      return static_cast<T>(1);

    if(i % 100 == 0)
      return static_cast<T>(2);

    return static_cast<T>(1);
  }
};

template<class T, typename U>
struct input_generator <T, sycl::minimum<U>> {
  T operator() (std::size_t i) const {
    return 1234 - (i % 567);
  }
};

template<class T, typename U>
struct input_generator <T, sycl::maximum<U>> {
  T operator() (std::size_t i) const {
    return 1234 + (i % 567);
  }
};

template<class T, typename U>
struct input_generator <T, sycl::bit_and<U>> {
  T operator() (std::size_t i) const {
    return static_cast<T>((2ull * static_cast<unsigned long long>(i)) | (1ull << (8 * sizeof(T) - 1)) | 1ull);
  }
};

template<class T, typename U>
struct input_generator <T, sycl::bit_or<U>> {
T operator() (std::size_t i) const {
  return static_cast<T>(i ? 2ull * static_cast<unsigned long long>(i) : (1ull << (8 * sizeof(T) - 1)));
}
};

template<class T, typename U>
struct input_generator <T, sycl::bit_xor<U>> {
  T operator() (std::size_t i) const {
    return static_cast<T>(17 * i);
  }
};

template<typename U>
struct input_generator <bool, sycl::logical_or<U>> {
  bool operator() (std::size_t i) const {
    return i == 0;
  }
};

template<typename U>
struct input_generator <bool, sycl::logical_and<U>> {
bool operator() (std::size_t i) const {
  return i != 0;
}
};


// Unknown to SYCL, so either an explicit identity must be provided or the slow non-identity reduction path is entered
template<typename T>
struct no_identity_maximum {
    T operator()(T a, T b) const { return a < b ? b : a ;}
};


template<class T, class BinaryOp, int Line>
class reduction_kernel;

template<class T, class BinaryOp>
void test_single_reduction(std::size_t input_size, std::size_t local_size, BinaryOp op) {
  sycl::queue q;

  const auto initialize_to_identity = sycl::property::reduction::initialize_to_identity{};

  auto input_gen = [&](std::size_t i){
    return input_generator<T,BinaryOp>{}(i);
  };

  test_scalar_reduction<T>(q, input_size, op, input_gen,
    [&](T* input, T* output, const sycl::property_list &reduction_prop_list){

      q.parallel_for<reduction_kernel<T,BinaryOp,__LINE__>>(
                  sycl::range<1>(input_size), 
                  sycl::reduction(output, op, reduction_prop_list),
                  [=](sycl::id<1> idx, auto& reducer){
        reducer.combine(input[idx[0]]);
      });

    });

  if(input_size % local_size == 0) {
    test_scalar_reduction<T>(q, input_size, op, input_gen,
      [&](T* input, T* output, const sycl::property_list &reduction_prop_list){

        q.parallel_for<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::nd_range(sycl::range{input_size}, 
                                     sycl::range{local_size}), 
                      sycl::reduction(output, op, reduction_prop_list),
                      [=](sycl::nd_item<1> idx, auto& reducer){
          reducer.combine(input[idx.get_global_linear_id()]);
        });

      });
    
    std::size_t num_groups = input_size / local_size;

    test_scalar_reduction<T>(q, input_size, op, input_gen,
      [&](T* input, T* output, const sycl::property_list &reduction_prop_list){

        q.submit([&](sycl::handler& cgh){
          cgh.parallel_for_work_group<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::range{num_groups}, sycl::range{local_size}, 
                      sycl::reduction(output, op, reduction_prop_list),
                      [=](sycl::group<1> grp, auto& reducer){
            grp.parallel_for_work_item([&](sycl::h_item<1> idx){
              reducer.combine(input[idx.get_global_id()[0]]);
            });
          });
        });
      });
   
    test_scalar_reduction<T>(q, input_size, op, input_gen,
      [&](T* input, T* output, const sycl::property_list &reduction_prop_list){

        q.parallel<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::range{num_groups}, sycl::range{local_size}, 
                      sycl::reduction(output, op, reduction_prop_list),
                      [=](sycl::group<1> grp, sycl::physical_item<1> pidx, 
                          auto& reducer){
          grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx){
            reducer.combine(input[idx.get_global_linear_id()]);
          });
        });
      });
    
  }
}


template<class T>
void test_two_reductions(std::size_t input_size, std::size_t local_size){
  sycl::queue q;

  const auto initialize_to_identity = sycl::property::reduction::initialize_to_identity{};

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

  q.parallel_for<reduction_kernel<T,class MultiOp,__LINE__>>(
              sycl::range<1>(input_size), 
              sycl::reduction(output0, sycl::plus<T>{}, initialize_to_identity),
              sycl::reduction(output1, sycl::multiplies<T>{}, initialize_to_identity),
              [=](sycl::id<1> idx, auto& add_reducer, auto& mul_reducer){
    add_reducer += input0[idx[0]];
    mul_reducer *= input1[idx[0]];
  });

  verify();

  if(input_size % local_size == 0) {

    q.parallel_for<reduction_kernel<T,class MultiOp,__LINE__>>(
                sycl::nd_range(sycl::range{input_size}, 
                               sycl::range{local_size}), 
                sycl::reduction(output0, sycl::plus<T>{}, initialize_to_identity),
                sycl::reduction(output1, sycl::multiplies<T>{}, initialize_to_identity),
                [=](sycl::nd_item<1> idx, auto& add_reducer, auto& mul_reducer){
      add_reducer += input0[idx.get_global_linear_id()];
      mul_reducer *= input1[idx.get_global_linear_id()];
    });

    verify();
    
    std::size_t num_groups = input_size / local_size;
    
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for_work_group<reduction_kernel<T,class MultiOp,__LINE__>>(
                  sycl::range{num_groups}, sycl::range{local_size},
                  sycl::reduction(output0, sycl::plus<T>{}, initialize_to_identity),
                  sycl::reduction(output1, sycl::multiplies<T>{}, initialize_to_identity),
                  [=](sycl::group<1> grp, auto& add_reducer, auto& mul_reducer){
        // On omp backend, this is executed in a pragma omp simd loop.
        // It seems clang generates incorrect code when doing two
        // reductions inside the same simd loop, so we need to split
        // them in two loops.
        grp.parallel_for_work_item([&](sycl::h_item<1> idx){
          mul_reducer *= input1[idx.get_global_id()[0]];
        });

        grp.parallel_for_work_item([&](sycl::h_item<1> idx){
          add_reducer += input0[idx.get_global_id()[0]];
        });
      });
    });

    verify();

    q.parallel<reduction_kernel<T,class MultiOp,__LINE__>>(
                sycl::range{num_groups}, sycl::range{local_size},
                sycl::reduction(output0, sycl::plus<T>{}, initialize_to_identity),
                sycl::reduction(output1, sycl::multiplies<T>{}, initialize_to_identity),
                [=](sycl::group<1> grp, sycl::physical_item<1> pidx,
                    auto& add_reducer, auto& mul_reducer){
      
      grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx){
        mul_reducer *= input1[idx.get_global_linear_id()];
      });

      grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx){
        add_reducer += input0[idx.get_global_linear_id()];
      });
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

using int_test_types = boost::mpl::list<char, unsigned int, int, long long>;

BOOST_AUTO_TEST_CASE_TEMPLATE(single_kernel_single_scalar_reduction, T, all_test_types) {
  test_single_reduction<T>(128, 128, sycl::plus<T>{});
  test_single_reduction<T>(128, 128, sycl::multiplies<T>{});
  test_single_reduction<T>(128, 128, sycl::minimum<T>{});
  test_single_reduction<T>(128, 128, sycl::maximum<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(single_kernel_single_scalar_int_reduction, T, int_test_types) {
  test_single_reduction<T>(128, 128, sycl::bit_or<>{});
  test_single_reduction<T>(128, 128, sycl::bit_and<>{});
  test_single_reduction<T>(128, 128, sycl::bit_xor<>{});
}

BOOST_AUTO_TEST_CASE(single_kernel_single_scalar_bool_reduction) {
  test_single_reduction<bool>(128, 128, sycl::logical_and<bool>{});
  test_single_reduction<bool>(128, 128, sycl::logical_or<bool>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(two_kernels_single_scalar_reduction, T, large_test_types) {
  test_single_reduction<T>(128*128, 128, sycl::plus<>{});
  test_single_reduction<T>(128*128, 128, sycl::multiplies<>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(three_kernels_single_scalar_reduction, T, very_large_test_types) {
  test_single_reduction<T>(64*64*64, 64, sycl::plus<T>{});
  test_single_reduction<T>(64*64*64, 64, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mismatching_group_size, T, large_test_types) {
  test_single_reduction<T>(1000, 128, sycl::plus<T>{});
  test_single_reduction<T>(1000, 128, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(oversized_group_size, T, all_test_types) {
  test_single_reduction<T>(20, 128, sycl::plus<T>{});
  test_single_reduction<T>(20, 128, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(oversized_group_size_int, T, int_test_types) {
  test_single_reduction<T>(20, 128, sycl::bit_or<T>{});
  test_single_reduction<T>(20, 128, sycl::bit_and<T>{});
  test_single_reduction<T>(20, 128, sycl::bit_xor<T>{});
}

BOOST_AUTO_TEST_CASE(oversized_group_size_bool) {
  test_single_reduction<bool>(20, 128, sycl::logical_or<bool>{});
  test_single_reduction<bool>(20, 128, sycl::logical_and<bool>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(two_reductions, T, large_test_types) {
  test_two_reductions<T>(128*128, 128);
}

BOOST_AUTO_TEST_CASE(buffer_reduction) {
  sycl::queue q;
  const auto initialize_to_identity = sycl::property::reduction::initialize_to_identity{};
  sycl::buffer<int> values_buff{1024};
  { 
    sycl::host_accessor a{values_buff};
    std::iota(a.get_pointer(), a.get_pointer() + 1024, 0);
  }
  
  sycl::buffer<int> sum_buff{1};
  sycl::buffer<int> max_buff{1};

  max_buff.get_host_access()[0] = 0;
  
  q.submit([&](sycl::handler &cgh) {
    auto values_acc = values_buff.get_access<sycl::access_mode::read>(
        cgh);

    auto sumReduction = sycl::reduction(sum_buff, cgh, sycl::plus<>(), initialize_to_identity);
    auto maxReduction = sycl::reduction(max_buff, cgh, sycl::maximum<int>());

    cgh.parallel_for(sycl::range<1>{1024}, sumReduction, maxReduction,
                     [=](sycl::id<1> idx, auto &sum, auto &max) {
                       sum += values_acc[idx];
                       max.combine(values_acc[idx]);
                     });
  });

  BOOST_CHECK(max_buff.get_host_access()[0] == 1023);
  BOOST_CHECK(sum_buff.get_host_access()[0] == 523776);
}

BOOST_AUTO_TEST_CASE(combine_none_or_multiple) {
  sycl::queue q;
  sycl::buffer<int> sum_buff{1};
  sycl::buffer<int> max_buff{1};
  sum_buff.get_host_access()[0] = 0;
  max_buff.get_host_access()[0] = 0;

  q.submit([&](sycl::handler &cgh) {
      auto sumReduction = sycl::reduction(sum_buff, cgh, sycl::plus<>());
      auto maxReduction = sycl::reduction(max_buff, cgh, sycl::maximum<int>()); // no identity

      cgh.parallel_for(sycl::range<1>{1024}, sumReduction, maxReduction,
          [=](sycl::id<1> idx, auto &sum, auto &max) {
              // Test both validity of 0 combines or > 1 combine per item
              for (int i = 0; i < static_cast<int>(idx.get(0)) % 3; ++i) {
                sum += idx[0];
                max.combine(idx[0]);
              }
          });
  });

  BOOST_CHECK(max_buff.get_host_access()[0] == 1022);
  BOOST_CHECK(sum_buff.get_host_access()[0] == 523435);
}

BOOST_AUTO_TEST_SUITE_END()