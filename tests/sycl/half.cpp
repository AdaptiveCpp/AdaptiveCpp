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


#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

BOOST_FIXTURE_TEST_SUITE(half_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(half_arithmetic) {
  
  auto tolerance = boost::test_tools::tolerance(0.0001);
  
  namespace s = cl::sycl;

  s::queue q;
  s::half h1 {1.0f};
  s::half h2 {2.0f};


  constexpr std::size_t num_tests = 4;
  s::buffer<s::half, 1> buff{s::range{num_tests}};
  q.submit([&](s::handler& cgh){
    s::accessor acc{buff, cgh};
    cgh.single_task([=](){
      acc[0] = h1 + h2;
      acc[1] = h1 / h2;
      acc[2] = acc[0] * acc[1];
      acc[3] = acc[1] - acc[0];
    });
  }).wait();
   
  float f1 = 1.0f;
  float f2 = 2.0f;
  float reference [num_tests];
  reference[0] = f1 + f2;
  reference[1] = f1 / f2;
  reference[2] = reference[0] * reference[1];
  reference[3] = reference[1] - reference[0];

  s::host_accessor hacc{buff};
  for(int i = 0; i < num_tests; ++i){
    float current_reference = reference[i];
    float current_computed = hacc[i];
    BOOST_TEST(current_reference == current_computed, tolerance);
  }
}

using half_test_types =
  boost::mpl::list<float, double,
                   int, unsigned int,
                   long, long long,
                   unsigned long, unsigned long long>;
BOOST_AUTO_TEST_CASE_TEMPLATE(half_operators, T, half_test_types) {
  namespace s = cl::sycl;
  auto tolerance = boost::test_tools::tolerance(0.0001);
  
  s::queue q;
  T a{1};
  s::half b{2.0f};

  constexpr std::size_t num_ops = 4;
  s::buffer<s::half, 1> buff_T{s::range{num_ops}}; // T as lhs operand
  s::buffer<s::half, 1> buff_half{s::range{num_ops}}; // half as lhs operand
  q.submit([&](s::handler& cgh){
    s::accessor acc_half{buff_half, cgh};
    s::accessor acc_T{buff_T, cgh};
    cgh.single_task([=](){      
      acc_T[0] = a + b;
      acc_T[1] = a - b;
      acc_T[2] = a * b;
      acc_T[3] = a / b;

      acc_half[0] = b + a;
      acc_half[1] = b - a;
      acc_half[2] = b * a;
      acc_half[3] = b / a;
    });
  }).wait();

  T f1{1};
  float f2 = 2.0f;
  
  float reference_T [num_ops]; // T as lhs operand
  reference_T[0] = f1 + f2;
  reference_T[1] = f1 - f2;
  reference_T[2] = f1 * f2;
  reference_T[3] = f1 / f2;

  float reference_half [num_ops]; // half as lhs operand
  reference_half[0] = f2 + f1;
  reference_half[1] = f2 - f1;
  reference_half[2] = f2 * f1;
  reference_half[3] = f2 / f1;
  
  s::host_accessor hacc_half{buff_half};
  s::host_accessor hacc_T{buff_T};
  for(int i = 0; i < num_ops; ++i){
    float current_reference_half = reference_half[i];
    float current_reference_T = reference_T[i];
    
    float current_computed_half = hacc_half[i];
    float current_computed_T = hacc_T[i];
    
    BOOST_TEST(current_reference_half == current_computed_half, tolerance);
    BOOST_TEST(current_reference_T == current_computed_T, tolerance);
  }
}


BOOST_AUTO_TEST_CASE(half_unary_operators){

  namespace s = cl::sycl; 
  namespace tt = boost::test_tools;

  // build inputs and allocate outputs

  s::queue queue;
  s::half input{1.0};

  constexpr std::size_t NUM_OPS = 2;
  s::buffer<s::half, 1> buff_half{s::range{NUM_OPS}}; 

  // run functions 

  queue.submit([&](s::handler& cgh) {
    s::accessor acc_half{buff_half, cgh};
    cgh.single_task([=](){
      acc_half[0] = +input;
      acc_half[1] = -input; 
    });
  }).wait();

  float f1 = 1.0f;
  float reference[NUM_OPS];
  reference[0] = +f1;
  reference[1] = -f1;
 
  // check results

  s::host_accessor hacc_half{buff_half};
  for(int i = 0; i < NUM_OPS; ++i){
    float current_reference = reference[i];
    float current_computed_half = hacc_half[i];
    BOOST_TEST(current_reference == current_computed_half, tt::tolerance(1e-4));
  }
}

BOOST_AUTO_TEST_SUITE_END()    
