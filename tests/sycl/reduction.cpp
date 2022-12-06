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

#include <boost/test/unit_test_suite.hpp>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(reduction_tests, reset_device_fixture)

auto tolerance = boost::test_tools::tolerance(0.001f);

template <class T, class Generator, class Handler, class BinaryOp, int Dim=1>
void test_scalar_reduction(sycl::queue &q, const T& identity,
                           sycl::range<Dim> num_elements,
                           BinaryOp op, Generator gen, Handler h) {
  // determine type of data array depending on dimension
  typedef typename std::conditional<
    Dim == 1,
    T*,
    typename std::conditional<Dim == 2, T**, T***>::type
    >::type ptrT;

  ptrT test_data;
  if constexpr (Dim == 1) {
    test_data = sycl::malloc_shared<T>(num_elements[0], q);
    for(std::size_t i = 0; i < num_elements[0]; ++i)
      test_data[i] = gen(i);
  }
  else if constexpr (Dim == 2) {
    test_data = sycl::malloc_shared<T*>(num_elements[0], q);
    for(std::size_t i = 0; i < num_elements[0]; ++i) {
      test_data[i] = sycl::malloc_shared<T>(num_elements[1], q);
      for (std::size_t j = 0; j < num_elements[1]; ++j) {
        test_data[i][j] = gen(i + j);
      }
    }
  } else if constexpr (Dim == 3) {
    test_data = sycl::malloc_shared<T**>(num_elements[0], q);
    for(std::size_t i = 0; i < num_elements[0]; ++i) {
      test_data[i] = sycl::malloc_shared<T*>(num_elements[1], q);
      for (std::size_t j = 0; j < num_elements[1]; ++j) {
        test_data[i][j] = sycl::malloc_shared<T>(num_elements[2], q);
        for (std::size_t k = 0; k < num_elements[2]; ++k) 
          test_data[i][j][k] = gen(i);
      }
    }
  }
  
  T* output_data = sycl::malloc_shared<T>(1, q);
  
  h(test_data, output_data);
  q.wait();

  T expected_result;

  if constexpr (Dim == 1) {
    expected_result =
      std::accumulate(test_data, test_data + num_elements[0], identity, op);
  } else if constexpr (Dim == 2) {
    expected_result = identity;
    for (std::size_t i = 0; i < num_elements[0]; ++i) {
      expected_result = op(expected_result,
                           std::accumulate(test_data[i],
                                           test_data[i] + num_elements[1],
                                           identity, op));
    }
  } else if constexpr (Dim == 3) {
    expected_result = identity;
    for (std::size_t i = 0; i < num_elements[0]; ++i) {
      for (std::size_t j = 0; j < num_elements[1]; ++j) {
        expected_result = op(expected_result,
                             std::accumulate(test_data[i][j],
                                             test_data[i][j] + num_elements[2],
                                             identity, op));
      }
    }
  }
  
  if constexpr(std::is_floating_point_v<T>) {
    BOOST_TEST(expected_result == *output_data, tolerance);
  } else {
    BOOST_TEST(expected_result == *output_data);
  }

  if constexpr (Dim == 1) {
    sycl::free(test_data, q);
  } else if constexpr (Dim == 2) {
    for (size_t i=0; i<num_elements[0]; ++i) {
      sycl::free(test_data[i], q);
    }
    sycl::free(test_data, q);
  } else if constexpr (Dim == 3) {
    for (size_t i=0; i<num_elements[0]; ++i) {
      for (size_t j=0; j<num_elements[1]; ++j) 
        sycl::free(test_data[i][j], q);
      sycl::free(test_data[i], q);
    }
    sycl::free(test_data, q);
  }

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

  test_scalar_reduction(q, identity, 
                        sycl::range{input_size}, op, input_gen,
    [&](T* input, T* output){

      q.parallel_for<reduction_kernel<T,BinaryOp,__LINE__>>(
                  sycl::range<1>(input_size), 
                  sycl::reduction(output, identity, op), 
                  [=](sycl::id<1> idx, auto& reducer){
        reducer.combine(input[idx[0]]);
      });

    });

  if(input_size % local_size == 0) {
    test_scalar_reduction(q, identity,
                          sycl::range{input_size}, op, input_gen,
      [&](T* input, T* output){

        q.parallel_for<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::nd_range(sycl::range{input_size}, 
                                     sycl::range{local_size}), 
                      sycl::reduction(output, identity, op), 
                      [=](sycl::nd_item<1> idx, auto& reducer){
          reducer.combine(input[idx.get_global_linear_id()]);
        });

      });
    
    std::size_t num_groups = input_size / local_size;

    test_scalar_reduction(q, identity,
                          sycl::range{input_size}, op, input_gen,
      [&](T* input, T* output){

        q.submit([&](sycl::handler& cgh){
          cgh.parallel_for_work_group<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::range{num_groups}, sycl::range{local_size}, 
                      sycl::reduction(output, identity, op), 
                      [=](sycl::group<1> grp, auto& reducer){
            grp.parallel_for_work_item([&](sycl::h_item<1> idx){
              reducer.combine(input[idx.get_global_id()[0]]);
            });
          });
        });
      });
   
    test_scalar_reduction(q, identity,
                          sycl::range{input_size}, op, input_gen,
      [&](T* input, T* output){

        q.parallel<reduction_kernel<T,BinaryOp,__LINE__>>(
                      sycl::range{num_groups}, sycl::range{local_size}, 
                      sycl::reduction(output, identity, op), 
                      [=](auto grp, auto& reducer){
          sycl::distribute_items(grp, [&](sycl::s_item<1> idx){
            reducer.combine(input[idx.get_global_linear_id()]);
          });
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

  q.parallel_for<reduction_kernel<T,class MultiOp,__LINE__>>(
              sycl::range<1>(input_size), 
              sycl::reduction(output0, T{0}, sycl::plus<T>{}),
              sycl::reduction(output1, T{1}, sycl::multiplies<T>{}), 
              [=](sycl::id<1> idx, auto& add_reducer, auto& mul_reducer){
    add_reducer += input0[idx[0]];
    mul_reducer *= input1[idx[0]];
  });

  verify();

  if(input_size % local_size == 0) {

    q.parallel_for<reduction_kernel<T,class MultiOp,__LINE__>>(
                sycl::nd_range(sycl::range{input_size},
                               sycl::range{local_size}),
                sycl::reduction(output0, T{0}, sycl::plus<T>{}),
                sycl::reduction(output1, T{1}, sycl::multiplies<T>{}),
                [=](sycl::nd_item<1> idx, auto& add_reducer, auto& mul_reducer){
      add_reducer += input0[idx.get_global_linear_id()];
#ifdef __HIPSYCL_USE_ACCELERATED_CPU__
      idx.barrier(); // workaround for omp simd failure as below.
#endif
      mul_reducer *= input1[idx.get_global_linear_id()];
    });

    verify();
    
    std::size_t num_groups = input_size / local_size;
    
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for_work_group<reduction_kernel<T,class MultiOp,__LINE__>>(
                  sycl::range{num_groups}, sycl::range{local_size},
                  sycl::reduction(output0, T{0}, sycl::plus<T>{}),
                  sycl::reduction(output1, T{1}, sycl::multiplies<T>{}), 
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

    q.parallel<reduction_kernel<T, class MultiOp, __LINE__>>(
        sycl::range{num_groups}, sycl::range{local_size},
        sycl::reduction(output0, T{0}, sycl::plus<T>{}),
        sycl::reduction(output1, T{1}, sycl::multiplies<T>{}),
        [=](auto grp, auto &add_reducer, auto &mul_reducer) {
          sycl::distribute_items(grp, [&](sycl::s_item<1> idx) {
            mul_reducer *= input1[idx.get_global_linear_id()];
          });

          sycl::distribute_items(
              grp, [&](sycl::s_item<1> idx) {
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

template<class T, class BinaryOp>
void test_2d_reduction(sycl::range<2> input_size, sycl::range<2> local_size, 
                       const T& identity, BinaryOp op){
  sycl::queue q;

  auto input_gen = [&](std::size_t i){
    return input_generator<T,BinaryOp>{}(i);
  };

  test_scalar_reduction(q, identity,
                        input_size, op, input_gen,
                        [&](T** input, T* output){
                          q.parallel_for<
                            reduction_kernel<T,BinaryOp,__LINE__>
                            >(input_size, 
                              sycl::reduction(output, identity, op),
                              [=](sycl::item<2> idx, auto& reducer){
                                auto idx0 = idx[0];
                                auto idx1 = idx[1];
                                reducer.combine(input[idx0][idx1]);
                              });
                        });

  if(input_size[0] % local_size[0] == 0 and input_size[1] % local_size[1] == 0) {
    test_scalar_reduction(q, identity,
                          input_size, op, input_gen,
      [&](T** input, T* output){

        q.parallel_for<
          reduction_kernel<T,BinaryOp,__LINE__>
          >(sycl::nd_range{input_size, local_size}, 
            sycl::reduction(output, identity, op), 
            [=](sycl::nd_item<2> idx, auto& reducer){
              auto idx0 = idx.get_global_id(0);
              auto idx1 = idx.get_global_id(1);
              reducer.combine(input[idx0][idx1]);
            });
      });    
  }
}

template<class T, class BinaryOp>
void test_3d_reduction(sycl::range<3> input_size, sycl::range<3> local_size, 
                       const T& identity, BinaryOp op){
  sycl::queue q;

  auto input_gen = [&](std::size_t i){
    return input_generator<T,BinaryOp>{}(i);
  };

  test_scalar_reduction(q, identity,
                        input_size, op, input_gen,
                        [&](T*** input, T* output){
                          q.parallel_for<
                            reduction_kernel<T,BinaryOp,__LINE__>
                            >(input_size, 
                              sycl::reduction(output, identity, op),
                              [=](sycl::item<3> idx, auto& reducer){
                                auto idx0 = idx[0];
                                auto idx1 = idx[1];
                                auto idx2 = idx[2];
                                reducer.combine(input[idx0][idx1][idx2]);
                              });
                        });

  if(input_size[0] % local_size[0] == 0 and
     input_size[1] % local_size[1] == 0 and
     input_size[2] % local_size[2] == 0) {
    test_scalar_reduction(q, identity,
                          input_size, op, input_gen,
      [&](T*** input, T* output){

        q.parallel_for<
          reduction_kernel<T,BinaryOp,__LINE__>
          >(sycl::nd_range{input_size, local_size}, 
            sycl::reduction(output, identity, op), 
            [=](sycl::nd_item<3> idx, auto& reducer){
              auto idx0 = idx.get_global_id(0);
              auto idx1 = idx.get_global_id(1);
              auto idx2 = idx.get_global_id(2);
              reducer.combine(input[idx0][idx1][idx2]);
            });
      });    
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

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_reduction, T, all_test_types) {
  test_2d_reduction<T>({64,16}, {64,16}, T{0}, sycl::plus<T>{});
  test_2d_reduction<T>({64,16}, {64,16}, T{1}, sycl::multiplies<T>{});

  test_3d_reduction<T>({16,8,4}, {16,16,4}, T{0}, sycl::plus<T>{});
  test_3d_reduction<T>({4,4,4}, {4,4,4}, T{1}, sycl::multiplies<T>{});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(local_size_one_reduction, T, all_test_types) {
  test_single_reduction<T>(128, 1, 0, sycl::plus<T>{});
  test_2d_reduction<T>({64,16}, {1,1}, T{0}, sycl::plus<T>{});
  test_3d_reduction<T>({16,4,4}, {1,1,1}, T{0}, sycl::plus<T>{});
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
  
  q.submit([&](sycl::handler
                   &cgh) {
    auto values_acc = values_buff.get_access<sycl::access_mode::read>(
        cgh);

    sycl::accessor sum_acc {sum_buff, cgh};
    sycl::accessor max_acc {max_buff, cgh};
    
    auto sumReduction = sycl::reduction(sum_acc, sycl::plus<int>());
    auto maxReduction = sycl::reduction(max_acc, sycl::maximum<int>());
    
    cgh.parallel_for(sycl::range<1>{1024}, sumReduction, maxReduction,
                     [=](sycl::id<1> idx, auto &sum, auto &max) {
                       sum += values_acc[idx];
                       max.combine(values_acc[idx]);
                     });
  });

  BOOST_CHECK(max_buff.get_host_access()[0] == 1023);
  BOOST_CHECK(sum_buff.get_host_access()[0] == 523776);
}

BOOST_AUTO_TEST_SUITE_END()
